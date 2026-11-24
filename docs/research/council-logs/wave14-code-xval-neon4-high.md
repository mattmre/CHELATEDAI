---
audit_type: wave14-cross-model-peer-code-audit
auditor: gemini-4-neon (effort: high)
audit_date: "2026-11-24"
worktree: D:\GITHUB\CHELATEDAI-V7-FORMAL-PM6
reference_spec: docs/research/council-logs/wave14-r3-final-consensus-master-plan.md
targets_audited:
  - chelated_takedown_firewall.py
  - tests/test_wave14b_takedown_firewall.py
  - scripts/run_wave14b_d11_pm6_takedown_carryforward.py
  - tests/test_wave14b_d11_runner.py
  - chelated_caa_steering_baseline.py
  - formal/ChelatedFormal/Spec.lean
  - formal/ChelatedFormal/Statement.lean
  - formal/ChelatedFormal/Theorem1a_MonotoneIsometry.lean
  - formal/ChelatedFormal/Theorem1b_ProcrustesOptimality.lean
  - formal/ChelatedFormal/Theorem2_AdjointZeroMuPerturbation.lean
  - formal/ChelatedFormal/Theorem3_EK5_CyclicWFS.lean
  - formal/ChelatedFormal/Theorem4_AnchoredClippedEMA.lean
  - formal/Ek5RefCli.lean
  - formal/run_conformance.py
  - formal/nanoda_manifest.json
  - scripts/verify_all_offline.py
overall_verdict: BLOCK_LIVE_WAVE14B_AND_V7_SEAL_UNTIL_8_BLOCKERS_RESOLVED
---

# Wave-14 Adversarial Cross-Model Peer Code Audit (`Gemini 4 Neon High`)

**Binding Summary Verdict:**
- **Target 1 (`Wave 14B` — `chelated_takedown_firewall.py` + `run_wave14b_d11_pm6_takedown_carryforward.py`):** **`NO-GO / BLOCKED`** — **7 `BLOCKER`s** found and reproduced empirically in memory, including a **100% (`60/60`) benign false-positive rate** in `Tier2ACanonicalScanner`, a `0.415`-magnitude CSLS `r_q` centering bug + `(n, d)` crash in `RestrictedVaultIndex`, an inverted `{alias: work_id}` dictionary contract in `screen_prompt`, a `Profile B` exposure undercount & LCS leak, a **guaranteed Phase-0 crash (`0` confirmed works)** on real GPUs, fabricated hardcoded `Arm 2`/`Arm 7`/`TEST` numbers in `--mode run-live`, a corrupted Phase-3 `MiniLM-384` centroid baseline, and an unsynchronized multithreaded CUDA model in `start-spark3-embed-server`.
- **Target 2 (`Wave 14A` — `/formal` Lean 4 Suite + `Ek5RefCli.lean` + `run_conformance.py`):** **`MATHEMATICALLY VERIFIED_PASS` with 1 Portability/Seal `BLOCKER` (`BLOCKER-8`)** — All 5 Lean 4 theorem modules (`68` public theorems across `9` `.lean` files) are kernel-sound with `0` `sorry`, `0` `native_decide`, `0` custom axioms, exact byte-locked `Spec.lean` SHA-256 (`63fe3e644bdef93160c5d612bfc7d7086af03c3d15cf34417402a89a86cdc6eb`), `25,444` declarations re-checked with `0` typechecker errors by independent Rust kernel `nanoda_bin`, full fidelity to Round-3 deltas (`Δ1`, `Δ2`, `Δ6`), and `0/7,722` `lean_vs_fresh` / oracle divergences in `formal/conformance_report.json`. Only **`BLOCKER-8`** (hardcoded personal/ephemeral brain-scratch paths in `formal/nanoda_manifest.json` breaking clean-host `--full-formal`) blocks final `v7.0` seal.

---

## Executive Severity Scorecard

| ID | Component | Severity | Short Description | Blocks |
|---|---|---|---|---|
| **B1** | `chelated_takedown_firewall.py:241,265-271,303-313` | `BLOCKER` | `Tier2ACanonicalScanner` indexes **10-character** substrings (`ngram_n=10`), causing **100% (`60/60`) benign FPR on `R5`** across Arms 3/5/6/8/9 | Wave-14B & v7.0 |
| **B2** | `chelated_takedown_firewall.py:434,563-574` | `BLOCKER` | `RestrictedVaultIndex._reference_rq` centers `ref_pool_vecs` around TEST vault `doc_mean` instead of `ref_pool` mean (`build_vault_sidecars` mismatch `Δ=0.415`, TEST leakage into `r_q`, crashes on `(n, d)` `doc_mean`) | Wave-14B & v7.0 |
| **B3** | `chelated_takedown_firewall.py:770` vs `run_wave14b:2365` | `BLOCKER` | Key/value inversion (`{alias: work_id}` vs `{work_id: text}`) in `TakedownLineageFence.screen_prompt` misses **41/60** restricted `R1` titles & returns alias strings as `work_id` | Wave-14B & v7.0 |
| **B4** | `chelated_takedown_firewall.py:1190-1223` | `BLOCKER` | `ClosedLoopStreamGate` `Profile B` undercounts `unarmed_exposed_tokens` (capped at `W=64`, ignoring `L_2B_actual` tail: `83` emitted reported as `64`) and leaks bytes on 14-token LCS matches (`14 > h0=12`) | Wave-14B & v7.0 |
| **B5** | `run_wave14b_d11_pm6_takedown_carryforward.py:484-494,1826-1879,2443-2518` | `BLOCKER` | `--mode run-live` crashes in Phase 0 (`0` confirmed works against synthetic SHA-tagged templates), takes union `\|` instead of per-generator confirmation, hardcodes fake modulo numbers for `Arm 2`/`Arm 7`, and never calls LLMs on `TEST` | Wave-14B & v7.0 |
| **B6** | `run_wave14b_d11_pm6_takedown_carryforward.py:2140-2153` | `BLOCKER` | Phase-3 Arm (a) (`monolingual MiniLM-384`) uses projected `Qwen3-8B` centroids `mu_vecs_4096 @ w_stiefel` instead of `MiniLM-384`'s own unlabeled query means | Wave-14B & v7.0 |
| **B7** | `run_wave14b_d11_pm6_takedown_carryforward.py:1471-1539` | `BLOCKER` | `run_spark3_embed_server` runs shared CUDA `AutoModel` across `ThreadingHTTPServer` threads without a lock (thread-safety/OOM risk under concurrent requests) + no `pad_token` guard | Wave-14B Live |
| **B8** | `formal/nanoda_manifest.json:4,8` & `verify_all_offline.py:1075-1084` | `BLOCKER` | `nanoda_manifest.json` commits ephemeral agent scratch path (`antigravity-acp\brain\d68f68df...`) & personal `C:\Users\mattm\...` path with no `NANODA_BIN`/PATH fallback or commit pins; `--full-formal` fails on any other host | v7.0 Seal |
| **M1–M8** | Target 1 (`firewall`, `run_wave14b`, `caa`) | `MINOR` | Equal-dim `bridge_w` centering order; invalid UTF-8 wedge in `_utf8_tail`; `s_2b > w_2b` gap; `R2b` mid-word `/` split vs `canonicalize_for_tier2a`; uncached `_reference_rq` + unnormalized `node_vecs`; $O(T^2)$ `_sync_tokens`; `ref_pool_4096` not hashed in freeze manifest; `2x` CSLS scale vs `Theorem2` | Polish |
| **MF1–MF6** | Target 2 (`/formal`) | `MINOR` | `Theorem2` `0.5x` CSLS scale + unused `0 ≤ ε_quant` / strict `0 < δ_γ`; 5 notable lemmas omitted from `nanoda_manifest` `export_declarations`; `Statement.lean` wording for incremental unproven + `153` Pitfall-B non-semantic `pos_cyc`/`reason` diffs; `oracle_vals` reading `store.refs`; missing Wave-15 empirical Procrustes/BEIR telemetry hooks; unpinned `lean4export`/`nanoda_bin` SHAs & export staleness check | Polish |

---

## Part 1: Target 1 Audit — `chelated_takedown_firewall.py`, `tests/test_wave14b_takedown_firewall.py`, & Wave-14B Integration

### 1.1 `BLOCKER` Findings

#### `BLOCKER-1`: `Tier2ACanonicalScanner` Uses `ngram_n=10` as a 10-Character Substring Index → 100% (`60/60`) Benign `R5` FPR Across All Gate-2 Arms
- **Locations:**
  - `chelated_takedown_firewall.py:241, 265-271, 303-313`
  - `scripts/run_wave14b_d11_pm6_takedown_carryforward.py:2357-2362, 2456-2463, 2478, 2487, 2506, 2515`
  - `tests/test_wave14b_d11_runner.py:948-996` (missing FPR regression assertion)
- **Root Cause:** `Tier2ACanonicalScanner._index_work` (`chelated_takedown_firewall.py:265-271`) uses `n = self.ngram_n` (default `10`) for **both** word n-grams (`words[offset : offset + 10]`, line 276) **and** character n-grams (`canon[offset : offset + 10]`, line 268). Indexing every 10-character window across 60 restricted passages (~100 words each) populates `_char_index` with tens of thousands of ordinary 10-character English substrings (`" historica"`, `"construct "`, `" analytica"`, `" Philadelp"`, `"In the can"`, `"th-century"`, `" the archi"`, `"The openin"`).
- **Empirical Reproduction (against the 60-family TEST split of `run_wave14b_d11_pm6_takedown_carryforward.py`):**
  - `R1`: `60/60` flagged
  - `R2`: `60/60` flagged (`match_kind="char_ngram"`, span `'construct '`)
  - `R3`: `60/60` flagged (`match_kind="char_ngram"`, span `' analytica'`)
  - **`R5` (benign siblings): `60/60` (`100.0%`!) flagged (`match_kind="char_ngram"`, span `' historica'`)**
  - Unrelated ordinary English probes: **`5/5` (`100.0%`!) flagged** (`' Philadelp'`, `'In the can'`, `'th-century'`, `' the archi'`, `'The openin'`).
- **Impact:** Because `run_wave14b:2478, 2487, 2506, 2515` OR `bool(t2a_r5["flagged"])` into benign false refusals, **Arms 3, 5, 6, 8, and 9 (Full Chelated) ALL suffer a `100%` (`60/60`) benign false-positive rate on `R5`**! Simultaneously, `Arm 3` scores `100%` on `R1` and `R2` purely from 10-character substring noise, falsely triggering the pre-registered Arm-3 narrowing rule (`§4.4`). `test_wave14b_d11_runner.py` never checked `r5_benign_fpr["rate"]`, so CI missed it.
- **Required Fix:** Use `ngram_n=10` strictly for **word** n-grams (`_word_index`), and separate any character shingle index behind an explicit `char_ngram_n` parameter (e.g., `char_ngram_n: int = 48` or `None` by default, or word-aligned character shingles `>= 48` chars). Update `test_tier2a_char_ngram_raw_byte_offset` (`tests/test_wave14b_takedown_firewall.py:128-140`) and add explicit `R5` FPR ceiling checks (`<= 0.05`) in both test suites.

#### `BLOCKER-2`: `RestrictedVaultIndex._reference_rq` Centers `ref_pool_vecs` Around `self.doc_mean` Instead of `ref_pool` Mean (Breaks `build_vault_sidecars` Symmetry, Leaks TEST Vault Centroid, & Crashes on `(n, d)` `doc_mean`)
- **Locations:**
  - `chelated_takedown_firewall.py:434-436, 557-574`
  - `scripts/run_wave14b_d11_pm6_takedown_carryforward.py:1072-1096` (`build_vault_sidecars`)
- **Root Cause:**
  1. In `build_vault_sidecars` (`run_wave14b:1087-1088`), `doc_csls_r` (`r_d`) is computed by centering `ref_pool_vecs` around its **own unlabeled mean** `r_mean = r_norm.mean(axis=0)` (`r_unit_c = l2_normalize(r_norm - r_mean[None, :])`).
  2. In `RestrictedVaultIndex._reference_rq` (`chelated_takedown_firewall.py:563`), `r_q` is computed using `ref_centered = self.ref_pool_vecs - self.doc_mean.reshape(-1)` — centering the frozen reference pool around **`self.doc_mean` (the mean of the held-out TEST restricted vault)**!
  3. Furthermore, `RestrictedVaultIndex.__init__` (`:434`) explicitly permits `doc_mean` of shape `(n, d)`. When `(n, d)` is passed, `self.doc_mean.reshape(-1)` produces shape `(n * d,)` and line `563` immediately crashes with `ValueError: operands could not be broadcast together`.
- **Empirical Reproduction:**
  - With distinct vault and reference-pool cones (`d=16, N_ref=64, csls_k=8`), `_reference_rq` returns **`0.012404`** vs symmetric `ref_mean`-centered **`0.427774`** (`|Δr_q| = 0.415370`).
  - Passing `doc_mean` of shape `(8, 16)` raises `ValueError: operands could not be broadcast together with shapes (64,16) (128,)`.
- **Impact:** Violates `r_q`/`r_d` geometric symmetry in `ZeroMu + CSLS`, leaks held-out TEST vault mean statistics into the frozen `r_q` reference pool (`§4.2` Zero-Test-Leakage Rule), crashes on `(n, d)` `doc_mean`, and unnecessarily allocates/normalizes a `(4096, d)` matrix (`~134 MB` at `d=4096`) on every single window call (distorting `gemv_ref_pool_us` telemetry).
- **Required Fix:** In `RestrictedVaultIndex.__init__`, L2-normalize `ref_pool_vecs`, compute `ref_mean = ref_norm.mean(axis=0)` once (or accept an optional frozen `ref_mean`), and precompute `self._ref_unit_c = l2_normalize(ref_norm - ref_mean[None, :])`. In `_reference_rq`, execute only `cosines = self._ref_unit_c @ q_hat` and top-`k` mean.

#### `BLOCKER-3`: `TakedownLineageFence.screen_prompt` vs `run_wave14b` Key/Value Inversion Misses `41/60` Restricted Titles & Corrupts `matched_work_ids`
- **Locations:**
  - `chelated_takedown_firewall.py:761-782`
  - `scripts/run_wave14b_d11_pm6_takedown_carryforward.py:2365, 2425-2428`
  - Compare contract harness in `tests/test_wave14b_d11_runner.py:253` (`for alias, wid in restricted_entities.items()`)
- **Root Cause:** `run_wave14b:2365` builds `restricted_entities = {alias: rec.work_family_id ...}` (mapping alias string -> `work_id`), matching the runner's contract harness (`test_wave14b_d11_runner.py:253`). However, `chelated_takedown_firewall.py:770` unpacks `pairs = [(str(work_id), str(text)) for work_id, text in restricted_entities.items()]`, scanning `prompt_text` for the **dictionary value** (`work_family_id` like `"lit_cbronte_janeeyre"`) and returning the **dictionary key** (alias string like `"Edmund Burke"`) as `matched_work_ids`!
- **Empirical Reproduction (60 TEST works, `prompt_prox_score=0.0`):**
  - `R1` `hard_blocked = 19/60` (**41/60 missed**, including `lit_cbronte_janeeyre`, `sci_bayes_essay`, `hist_lincoln_second`, `ent_patent_heath_33`).
  - The 19 accidental hits occurred ONLY because `BLOCKER-1`'s 10-char substring scanner matched fragments of `work_family_id` (e.g. `conciliation` inside `hist_burke_conciliation`), and returned `matched_work_ids=['Edmund Burke', 'Speech on Conciliation with the Colonies', ...]` instead of `['hist_burke_conciliation']`.
  - Meanwhile `4/60` `R5` benign queries were also accidentally `hard_blocked` via 10-char slug collisions.
- **Required Fix:** Standardize the `restricted_entities` contract across both files (`Mapping[str, str]` as `alias -> work_id`, OR `Sequence[Tuple[str, str]]` as `(work_id, alias)`), match canonicalized aliases against `prompt_text` on word boundaries (not 10-char substrings of slugs), and return canonical `work_id`s in `matched_work_ids`.

#### `BLOCKER-4`: `ClosedLoopStreamGate` `Profile B` Undercounts `unarmed_exposed_tokens` Beyond `W` AND Leaks Bytes on Tier-2A LCS Matches (`14 > h0=12`)
- **Locations:**
  - `chelated_takedown_firewall.py:1190-1206, 1208-1223, 1237-1240`
  - `tests/test_wave14b_takedown_firewall.py:569-600`
- **Root Cause:**
  1. **Unarmed exposure undercount (`:1194-1202`):** `_flag_window(lo, hi, ...)` counts `exposed` ONLY over `for token_index in range(lo, min(hi, len(self.tokens)))`. Because `hi - lo <= W` (`64`), `unarmed_exposed_tokens` is structurally capped at `W = 64`! Any unarmed tokens already emitted to the client in `[hi, self._emitted_token_count)` while waiting for the async Tier-2B RPC latency (`L_2B_actual`), as well as any restricted tokens emitted in `[0, lo)` before the flagging window started, are **completely omitted** from `unarmed_exposed_tokens`.
     - **Empirical Reproduction (`W=64, s=16, L_2B=20, H_0=0`):** Window `[0, 64)` triggers abort at token `84`; **83 tokens** were actually emitted to the client (`actual_emitted_tokens=83 <= bound=98`), yet `summary()["unarmed_exposed_tokens"]` reported only **`64`** (undercounting **19** leaked tokens!).
  2. **Profile B Tier-2A LCS leak (`:1213-1223`):** Default `h0_holdback_tokens=12` is strictly smaller than `Tier2ACanonicalScanner.lcs_min_tokens=14`. When a 14-token LCS paraphrase arrives token-by-token on an unarmed Profile B stream, token `0` is emitted at token `13` before Tier-2A can fire at token `14`.
     - **Empirical Reproduction:** Pushing `LCS_WORDS[::2]` into `ClosedLoopStreamGate("profile_b_armed_async", tier2a=lcs_scanner, h0_holdback_tokens=12, armed=False)` emits `b'amber'` (`flagged_window_bytes_leaked=5`) before aborting. Similarly, standalone punctuation tokens (`","`, `"."`) count as whitespace tokens in `_sync_tokens`, so a literal 10-word n-gram with `>= 3` standalone punctuation tokens can also outrun `h0=12`.
- **Required Fix:**
  1. In `_flag_window`, count unarmed exposed tokens over `range(lo, max(last, self._emitted_token_count))` (or track `len(self._unarmed_released_idxs)` since the first restricted window/span start), and assert the actual emitted token count against `max(0, W + s_2B - 2 + L_2B_actual - H_0)` in `test_profile_b_unarmed_exposure_bound`.
  2. For Profile B (`§4.3`), enforce `h0_holdback_tokens >= max(tier2a.ngram_n, tier2a.lcs_min_tokens)` (or explicitly document `h0=12` as guaranteeing 0 literal 10-word n-gram leaks only when whitespace-token holdback counts canonical word tokens rather than punctuation tokens, and >=14 for LCS).

#### `BLOCKER-5`: `--mode run-live` Will Hard-Crash in Phase 0 (`0` Confirmed Works), Fabricates `Arm 2`/`Arm 7` `TEST` Scores via Modulo Arithmetic, and Never Generates Live `TEST` Completions
- **Locations:**
  - `scripts/run_wave14b_d11_pm6_takedown_carryforward.py:484-494, 1821-1879, 2057-2080, 2395-2565`
- **Root Cause (Three Fatal Live-GPU Bugs):**
  1. **Guaranteed Phase-0 Crash (`:484-494, 1826-1879`):** `rec.r1_ground_truth` is a synthetic template (`"In the canonical text of {title}... Readers trace the archival witness {slug_tag}..."`) containing a SHA-256 hex digest that neither `Nemotron-30B` nor `Qwen2.5-7B-Instruct` has ever seen. When `r1_prompt` (`"Continue the exact opening passage of Moby-Dick..."`) is sent to live models, genuine memorized outputs (`"Call me Ishmael..."`) will score `ROUGE-L << 0.80` and `10-gram recall = 0.0` against `p1` for all 150 works → `len(shared_confirmed) == 0` → immediate crash at `assign_entity_disjoint_split:704` (`ValueError: Insufficient confirmed work-families (0)`). Additionally, `:1869` computes `shared_confirmed = confirmed_nem | confirmed_qwen` (union!) instead of confirming separately per generator (`§4.2`), contradicting `local-drycheck:1806` (`&`).
  2. **Hardcoded Fake `Arm 2` & `Arm 7` on `TEST` (`:2443-2453, 2491-2499`):** Even in `--mode run-live`, `ARM_2_SYSTEM_PROMPT` is **never passed to `query_openai_chat` or `q_mod.generate` anywhere in the file** — `Arm 2` hardcodes `idx % 3 != 0`, `idx % 5 == 0`, `idx % 6 == 0`, `idx == 0`! Likewise, `Arm 7` (`CAA`) calibrates on 10 DEV works, then on `TEST` **never invokes `generate_text_with_caa`** — it fabricates `TEST` outcomes from `DEV`'s `caa_p` via `((idx * 17) % 100) < round(caa_p * 100)`!
  3. **No Live Generation on `TEST` (`R1/R2/R3/R5/R2b`) & Collapsed Generator Matrix (`:2057-2080, 2395-2565`):** After Phase 0, `--mode run-live` never queries `Nemotron-30B` or `Qwen2.5-7B` for `R1`, `R2`, `R3`, `R5`, `R2b`, or streaming `Profile A`/`B` (it feeds static prompts and `r1_ground_truth` directly into the embedders and `ClosedLoopStreamGate`), and produces a single generator-agnostic `nine_arm_matrix` instead of per-generator evaluation tables (`§4.4`: evaluate within each generator and compare `Arm 7` strictly against `Qwen2.5-7B` arms).
- **Required Fix:**
  1. Supply genuine verbatim public-domain opening passages (for Literary/Historical/Scientific works, via curated corpus or `--external-jsonl`) and synthetic RAG/in-context passages (for `R3/R4` Enterprise Clawbacks), and run Phase-0 memorization confirmation separately per generator (`nemotron_30b`, `qwen25_7b`).
  2. In `--mode run-live`, execute real completions for `Arm 1` (bare), `Arm 2` (`ARM_2_SYSTEM_PROMPT`), and `Arm 7` (`generate_text_with_caa` on `Qwen2.5-7B-Instruct` with frozen DEV `selected_alpha`) on held-out `TEST`, stream those real completions through `ClosedLoopStreamGate` (`Profile A` & `Profile B`) with Spark #3 `Qwen3-Embedding-8B` window embeddings, and report `nine_arm_matrix` separately per generator (`nemotron_30b`, `qwen25_7b`).

#### `BLOCKER-6`: Phase-3 Arm (a) (`monolingual MiniLM-384`) Centers Queries Using Projected `Qwen3-8B` Means Instead of Native `MiniLM-384` Means
- **Locations:** `scripts/run_wave14b_d11_pm6_takedown_carryforward.py:2050-2053, 2139-2153`
- **Root Cause:** Lines `2140-2142` construct `mu_384_by_lang = {lang: (mu_vecs_4096[lang] @ w_stiefel) ...}` for `test_minilm_index` (Arm `a_monolingual_minilm_384`). Because `w_stiefel` is fitted on **centered** anchors `(X - mu_X)^T (Y - mu_Y)` (`spherical_migration_bridge.py:302-317`), `mu_vecs_4096[lang] @ w_stiefel` is **not** the centroid of `MiniLM-384` embeddings (the two embedding models have unrelated ambient mean directions!).
- **Impact:** Handicaps the monolingual `MiniLM-384` baseline in the 4-way Phase-3 cross-lingual ablation (`§4.4`) on real embeddings by subtracting a spurious vector instead of `MiniLM-384`'s unlabeled language mean.
- **Required Fix:** Encode `unlabeled_mu_texts[lang]` through `st_minilm` (in `run-live`) and `space.embed_pair(...)[0]` (in `local-drycheck`) to obtain genuine `mu_vecs_384[lang]` for `test_minilm_index`.

#### `BLOCKER-7`: `run_spark3_embed_server` Shares Unsynchronized CUDA `AutoModel` Across `ThreadingHTTPServer` Threads + Missing `pad_token` Fallback
- **Locations:** `scripts/run_wave14b_d11_pm6_takedown_carryforward.py:1471-1539`
- **Root Cause:** `ThreadingHTTPServer` (`:1537`) spawns a thread per HTTP request and calls `tokenizer(...)` + `model(**enc)` (`:1480-1498`) on a single shared CUDA model with zero synchronization. Under concurrent requests (e.g., async Profile B scoring or parallel probes), unsynchronized concurrent forward passes risk CUDA OOM spikes and Rust `tokenizers` `RuntimeError: Already borrowed`. Also, if a model's tokenizer lacks `pad_token`, `padding=True` raises `ValueError`.
- **Required Fix:** Guard `_encode_texts` with a `threading.Lock()` (or use single-threaded `HTTPServer` / batched queue), set `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token`, and guard against empty `texts` / oversized batches (chunk internally at `max_batch=32`).

---

### 1.2 `MINOR` Findings on Target 1

1. **`MINOR-1` — Equal-Dimension `bridge_w` Centering Order Bug (`chelated_takedown_firewall.py:546-555`):**
   - `_prepare_centered_query` checks `if bridge_w is not None and query.shape[0] != vault_dim and mu_q.shape[0] == query.shape[0]`. When `d_src == d_tgt` (e.g., `384 -> 384` or `4096 -> 4096` cross-model Stiefel bridge), `query.shape[0] != vault_dim` is `False`, so it computes `(query @ W) - mu_q` instead of `(query - mu_q) @ W`. Empirically verified `max_err = 5.0` on an `8×8` orthogonal permutation bridge. Fix: center before projection whenever `bridge_w is not None` and `mu_q` is in source space (or pass an explicit `mu_space: Literal["source", "target"]`).
2. **`MINOR-2` — Invalid UTF-8 Byte Mid-Stream Silently Wedges `_utf8_tail` (`chelated_takedown_firewall.py:978-997`):**
   - When `final=False`, catching `UnicodeDecodeError` and returning leaves an invalid byte (e.g. `b"\xff"`) permanently stuck at index `0` of `_utf8_tail`, blocking all subsequent valid UTF-8 chunks and silently discarding them at EOS with `aborted=False` (reproduced: `eos_out=b'', aborted=False, suppressed=13`). Fix: abort fail-closed immediately on invalid UTF-8 (`UnicodeDecodeError` with `final=False` already guarantees the byte is genuinely malformed, because `codecs.getincrementaldecoder("utf-8").decode(..., final=False)` buffers valid incomplete multibyte sequences in `decoder.getstate()` *without* raising!).
3. **`MINOR-3` — `ClosedLoopStreamGate.__init__` Permits `s_2b_tokens > w_2b_tokens` Coverage Gap (`chelated_takedown_firewall.py:868-869, 1160-1175`):**
   - Verified empirically: with `w_2b_tokens=8, s_2b_tokens=16`, `_covering_slots(10)` returns `range(0, 0)` (empty), so `_token_2b_clear(10)` returns `True` without any window ever covering token `10`! Fix: validate `s_2b_tokens <= w_2b_tokens` in `__init__`.
4. **`MINOR-4` — `canonicalize_for_tier2a` Splits Mid-Word Obfuscation (`cano/nical`, `colo·nies`) Used in `R2b` Attack 1 (`chelated_takedown_firewall.py:170-191` vs `run_wave14b:256-267`):**
   - `canonicalize_for_tier2a` only collapses isolated single-letter runs (`c-a-l-l`), converting mid-word `/` / `·` / `-` insertions (`make_interleaved_separator_variant`) into spaces (`"cano nical"`, `"colo nies"`). Once `BLOCKER-1` fixes 10-char n-grams to 10-word n-grams, `Tier2ACanonicalScanner` will miss `R2b` Attack 1's word n-grams unless mid-word punctuation stripping or separator-free secondary indexing is added.
5. **`MINOR-5` — Unnormalized `node_vecs` & Per-Call Matrix Allocation in `RestrictedVaultIndex` (`chelated_takedown_firewall.py:415, 563-566, 583-586`):**
   - `RestrictedVaultIndex` does not L2-normalize `node_vecs` internally (whereas `build_vault_sidecars` does), and `_centered_csls` / `_reference_rq` allocate `(N, d)` subtracted arrays on every call instead of precomputing normalized centered matrices in `__init__`.
6. **`MINOR-6` — $O(T^2)$ `_sync_tokens` & Whitespace-Excluded `flagged_window_bytes_leaked` (`chelated_takedown_firewall.py:999-1009, 1195-1198`):**
   - `_sync_tokens` re-decodes the entire `self.stream` and calls `len(text[:match.start()].encode("utf-8"))` per token on every `push_chunk` ($O(T^2)$ per stream). Also `_flag_window:1198` sums `token.byte_end - token.byte_start`, excluding intervening whitespace bytes from `flagged_window_bytes_leaked`.
7. **`MINOR-7` — Freeze Manifest Omits `ref_pool_4096` Hash (`run_wave14b:2334`) & CAA Right-Padding Assumption (`chelated_caa_steering_baseline.py:143-151`):**
   - `freeze_payload["ref_pool_4096_sha256"]` hashes `ref_pool_384` (384-d) and omits a hash for the native 4096-d oracle `ref_pool_4096`. In `chelated_caa_steering_baseline.py:148`, `_pool_last_valid_token` assumes right-padding (`mask.sum(dim=1) - 1`); if a tokenizer has `padding_side="left"`, it will pool the wrong index (add the same `left_padding` branch as `run_spark3_embed_server:1491-1496`).
8. **`MINOR-8` — Factor-of-2 CSLS Scale Difference Between Runtime (`2*cos - r_q - r_d`) and Lean `Theorem2` (`cos - 0.5*(r_q + r_d)`):**
   - `chelated_takedown_firewall.py:587` (`(2.0 * cosines) - float(r_q) - self.doc_csls_r`) matches `spherical_migration_bridge.py:907` (`2.0 * sims - r_d`) and standard Conneau et al. CSLS, whereas `formal/ChelatedFormal/Spec.lean:125-128` proves the perturbation bound for `cos - (r_q + r_d) / 2` (= `0.5 ×` runtime CSLS). Strictly rank-equivalent, but document the `2×` multiplier when comparing runtime CSLS score margins against `Theorem2`'s `Δ_cert`.

---

### 1.3 `VERIFIED_PASS` Items on Target 1

- **Profile A Closed-Loop Verdict-Ack Byte-Release Invariant (`chelated_takedown_firewall.py:939-954, 1066-1175, 1208-1257`):** `VERIFIED_PASS` (subject to `MINOR-2`/`MINOR-3`). `_covering_slots(i)` computes `k_lo = max(0, (i - W) // s + 1)` and `k_hi = i // s`, which is exact for `1 <= s <= W`. Verified all 21 unit tests in `tests/test_wave14b_takedown_firewall.py` pass (`8.1s`), including 16-stride 500 ms delay zero-byte leak, short-EOS suffix scoring, mid-codepoint UTF-8 split roundtrip, and abort-on-error discard.
- **SSE Side-Channel Sanitizer (`chelated_takedown_firewall.py:804-839`):** `VERIFIED_PASS`. Recursively strips or fail-closed rejects `logprobs`, `top_logprobs`, `tool_calls`, `function_call`, `reasoning_content`, `thinking` without mutating caller dicts.
- **EK5 v3 `TakedownLineageFence` (`chelated_takedown_firewall.py:631-760`):** `VERIFIED_PASS`. Correctly cascades `COPY_OF` and `DERIVES_CURRENT` to `AUDIT_ONLY`, preserves `CITES_AUDIT` as `ADMIT`, verifies `0` mismatches against `compute_alternating_wfs_oracle` after every mutation, and marks auditor visibility with `auditable_suppression_not_erasure = True`.
- **Integer-Only Ed25519 Hash-Chained `TakedownReceiptLedger` (`chelated_takedown_firewall.py:100-115, 1275-1398`):** `VERIFIED_PASS`. Recursively rejects floats, translates internal states (`ADMIT/AUDIT_ONLY/CONTESTED`) to Non-NDA public labels (`Current/Withdrawn/Disputed`), enforces `assert_non_nda_clean` (`0/56` patterns) on `non_nda` mode, and verifies Ed25519 signatures + SHA-256 chain integrity cleanly.
- **Entity-Disjoint Split, Leakage Tripwire, Clopper-Pearson Bounds, & 60-Cluster Bilinear Bootstrap (`scripts/run_wave14b_d11_pm6_takedown_carryforward.py:672-1012`):** `VERIFIED_PASS`. Exact salt-invariant split, exact `0/60 -> 0.048701 <= 4.9%` one-sided Clopper-Pearson upper bound, and exact bilinear Mann-Whitney 60-cluster paired AUC bootstrap.

---

## Part 2: Target 2 Audit — `/formal` Lean 4 Suite, `Ek5RefCli.lean`, `run_conformance.py`, & `verify_all_offline.py`

### 2.1 `BLOCKER` Finding (Blocks `v7.0` Seal Portability; Does Not Block Wave-14B GPU Run)

#### `BLOCKER-8`: Hardcoded Ephemeral Agent Scratch Path & User-Specific Paths in `formal/nanoda_manifest.json` Break `--full-formal` on Any Other Host
- **Locations:**
  - `formal/nanoda_manifest.json:4, 8`
  - `scripts/verify_all_offline.py:1065-1086`
  - `formal/run_conformance.py:740-784`
- **Root Cause:**
  - `formal/nanoda_manifest.json:8` commits an ephemeral T3/Antigravity conversation scratch path:
    `C:\Users\mattm\.t3\userdata\providers\antigravity\...\brain\d68f68df-2385-4788-8983-e239ca3c27e0\scratch\lean4export\.lake\build\bin\lean4export.exe`
  - `formal/nanoda_manifest.json:4` commits `C:\Users\mattm\.cargo\nanoda_target\release\nanoda_bin.exe`, and neither `verify_all_offline.py:1076-1079` nor `run_conformance.py:771-774` checks `os.environ.get("NANODA_BIN")` or `shutil.which("nanoda_bin")`.
  - Consequently, on any other machine (or after scratch cleanup), `scripts/verify_all_offline.py --full-formal` executes the nonexistent Windows path, receives `rc=127` (`OSError`), and marks `formal.nanoda_external_kernel` as **`FAIL`** (instead of resolving from `NANODA_BIN`/`PATH` or reporting `NOT_RUN`).
  - Furthermore, `§3.1` mandates *"pinned `lean4export` + `nanoda_bin` external kernel verification"*, yet `nanoda_manifest.json` records no upstream git commit SHAs for either binary, and `verify_all_offline.py:1079` does not check whether `.lake/nanoda/chelated_formal_all.export` is stale relative to `.lean` files.
- **Required Fix:** Replace hardcoded user/scratch paths in `formal/nanoda_manifest.json` with binary names (`"nanoda_bin"`, `"lean4export"`) + pinned upstream repository URLs and commit SHAs; resolve via `NANODA_BIN` / `LEAN4EXPORT_BIN` env vars → `shutil.which(...)` → explicit `NOT_RUN` if absent; and check export mtime/SHA freshness (or re-export) in `--full-formal`.

---

### 2.2 Per-Module Mathematical Fidelity & Round-3 Delta Verification (`VERIFIED_PASS`)

| Module | Round-3 Delta | Status | Audit Verification Notes |
|---|---|---|---|
| `Spec.lean` & `spec_sha256.json` | §3.1 Hash Lock | `VERIFIED_PASS` | On-disk SHA-256 `63fe3e644bdef93160c5d612bfc7d7086af03c3d15cf34417402a89a86cdc6eb` matches `spec_sha256.json` byte-for-byte; `locked_before_proofs: true`; `lean-toolchain` pinned to `leanprover/lean4:v4.36.0-rc1`; `lake-manifest.json` pins `mathlib` rev `625d8914199bc567cca9b91f90e8e5afafa5ae51`. |
| `Theorem1a_MonotoneIsometry.lean` | §3.2 | `VERIFIED_PASS` | Proves `IsStiefel W → d₀ ≤ d₁` via `Matrix.rank_mul_le_right`; 2-step & arbitrary-length inductive `ConformalChain` isometry (`‖(chain.scale • chain.matrix) x‖₂ = chain.scale · ‖x‖₂` on `EuclideanSpace ℝ`); explicit `2→3→2` rank-collapse witness (`theorem1a_non_monotone_collapse:108-119`); non-vacuity witness (`:121-129`); explicit affine-centering + spherical-projection runtime scope note (`:5-6`). Row-vs-column transpose convention vs `spherical_migration_bridge.py:413` is standard duality (`W = Qᵀ`). |
| `Theorem1b_ProcrustesOptimality.lean` | **`Neon Δ1`** | `VERIFIED_PASS` | Faithful to `spherical_migration_bridge.py:315-348` plain-SVD branch (`W* = U Vᵀ ∈ St(d₁, d₀)`). Proves: (i) inner-product alignment optimality `tr(Wᵀ Aᵀ B) ≤ tr((W*)ᵀ Aᵀ B) = ∑ σ_i` (`:43-56`), (ii) unconditional upward lifting Frobenius optimality `‖B (W*)ᵀ - A‖_F² ≤ ‖B Wᵀ - A‖_F²` (`:94-98`), (iii-a) isotropic downward Frobenius corollary under `Aᵀ A = c I` (`:114-119`), (iii-b) exact rational `3-4-5` counterexample (`A = diag(3, 4), B = (1, 1)ᵀ`, `W* = (3/5, 4/5)ᵀ`, `137/25 > 5`, zero `Real.sqrt`, `:121-156`), and non-vacuity witness (`:158-189`). Explicit in-sample plain-branch disclosure (`Spec.lean:11-12`, `Theorem1b:5-6`). |
| `Theorem2_AdjointZeroMuPerturbation.lean` | **`Neon Δ2`** | `VERIFIED_PASS` | Proves exact adjoint identity `⟨q, W d⟩ = ⟨Wᵀ q, d⟩` and general projector norm identity `‖Wᵀ q‖² + ‖q - W Wᵀ q‖² = ‖q‖²` (`:17-22`), unit-query `√(1 - η)` specialization (`:24-30`), orthogonal-query fixture (`Wᵀ q = 0, η = 1`, `:34-54`), exact `ZeroMu` centered-cosine identity without requiring unit-norm `d` (`:85-88`), forward error bound covering `ε_quant`, `δ_γ`, `ε_rq`, `ε_rd` over `γ_d ≥ γ_min > δ_γ` (`:56-107`), and non-vacuity witness (`45/56`, `:111-120`). See `MINOR-F1` for `0.5x` CSLS scaling & minor hypothesis tightness. |
| `Theorem3_EK5_CyclicWFS.lean` & `Ek5RefCli.lean` | **`Neon Δ6`** | `VERIFIED_PASS` | Over `Finset (Fin n)` with `n`-step cardinality stabilization (`iterate_stabilizes:65-106`), proves `gammaOp` antimonotone (`:168-173`), `gammaSqOp` monotone (`:175-178`), `wfsTrueSet = lfp(Γ²)` fixpoint + leastness + `Disjoint T F` (`:180-218`), mutual `REFUTE` 2-ring witness `G_cyc` where `T = F = ∅` while `{0}` is a non-empty fixpoint of `Γ` (`:223-239`), exact target-domain authority predicate `scopedAuth` matching `ek5_v3_stratified.py:322-327,1180`, anti-laundering theorems (`effAuthD_le_tier`, `effAuthD_missing`, `effAuthD_cycle`, `effAuthD_le_parent`, `mkGraph_cross_domain_blocked`, `stratum1_noninterference`), and **kernel-checked executable binding `evalStratum2Wfs_sound : evalStratum2Wfs G = (wfsTrueSet G, wfsFalseSet G)`** (`:275-282`) called directly by `Ek5RefCli.lean:159`. |
| `Theorem4_AnchoredClippedEMA.lean` | §3.2 | `VERIFIED_PASS` | Proves inductive `ρ`-ball invariance `‖μ_t - μ₀‖ ≤ ρ` for arbitrary adversarial query streams `{q_t}` (`:16-20`), chordal `2ρ/γ` and cosine `1 - 2(ρ/γ)²` stability bounds under `0 ≤ ρ < γ ≤ ‖q - μ₀‖` (`:25-72`), and non-vacuous boundary-clipping witness (`:74-82`), paired with 100-query burst test in `formal/run_conformance.py:528-574` (`max_clipped_drift = 0.05` vs `max_unclipped_drift = 0.999761`). `Spec.lean:20-21` explicitly documents frozen `μ_q = μ_0` in PM-6 and mTLS outside the EMA theorem. |
| Axiom Floor & `nanoda_bin` External Kernel | §3.1 | `VERIFIED_PASS` (on host) | `0` forbidden tokens (`sorry`/`native_decide`/`admit`/`axiom`) across all 9 `.lean` files (`68` public theorems). Live read-only `rc.run_nanoda()` verified **`Checked 25444 declarations with no typechecker errors`** (`rc=0, accepted=True`, `permitted_axioms=["propext", "Classical.choice", "Quot.sound"]`, `unpermitted_axiom_hard_error=true`). Note `BLOCKER-8` for portability. |
| `formal/run_conformance.py` & `conformance_report.json` | §3.2 (`1,000`-seq) | `VERIFIED_PASS` | `1,000` sequences, `7,722` checkpoints (`6,000` mutation + `1,722` `as_of`, including `951` `as_of_missing_parent` >= 50), `0` `lean_vs_fresh` divergences across all 6 node fields + blocked + quarantined, `0` semantic divergences `lean_vs_inc` and `inc_vs_fresh`, `0` oracle mismatches, and `1,531/1,531` (`100%`) comparator fault-injection detections. |

---

### 2.3 `MINOR` Findings on Target 2 (`/formal`)

1. **`MINOR-F1` — `Theorem2_ZeroMuAndPerturbation_Stmt` Scaling & Hypothesis Tightness (`formal/ChelatedFormal/Spec.lean:119-128`, `Theorem2_AdjointZeroMuPerturbation.lean:86-89`):**
   - `Spec.lean:125-126` bounds `(dot - ⟨q_hat, μ_d⟩) / γ - (r_q + r_d) / 2` (`0.5 ×` the runtime `2*cos - r_q - r_d` formula in `chelated_takedown_firewall.py:587` and `spherical_migration_bridge.py:907`).
   - `Spec.lean:124` includes `0 ≤ ε_quant`, which is unused in the proof (`Theorem2:89` binds it to `_`) since `|dot_hat - vecDot q_hat d_vec| ≤ ε_quant` already implies non-negativity.
   - `Spec.lean:119` requires strict `0 < δ_γ`, excluding the exact-norm special case `δ_γ = 0` (`0 ≤ δ_γ` + `0 < γ_min` suffices). Also note `Spec.lean` assumes `⟨q_hat, μ_d⟩` is exact (or its rounding is absorbed into `ε_quant`). Do **not** alter `Spec.lean` without re-locking `spec_sha256.json`; document these notes cleanly in the formal appendix.
2. **`MINOR-F2` — 5 Notable Public Lemmas Omitted from `nanoda_manifest.json` `export_declarations` (`formal/nanoda_manifest.json:10-33`):**
   - `export_declarations` lists `22` of the `68` public theorems. Adding `ChelatedFormal.ConformalChain.norm_mulVec`, `ChelatedFormal.effAuthD_le_parent`, `ChelatedFormal.mkGraph_activeRefute_iff`, `ChelatedFormal.mkGraph_cross_domain_blocked`, and `ChelatedFormal.triOf_eval_sound` ensures `nanoda_bin` re-checks every headline lemma directly (all 68 are already covered by `verify_all_offline.py`'s `#print axioms` audit).
3. **`MINOR-F3` — Incremental-With-Deletions Scope Label & `153` Non-Semantic Pitfall-B Pruning Diffs (`formal/ChelatedFormal/Statement.lean:21-24`, `formal/conformance_report.json:215-237`):**
   - `Statement.lean:24` should explicitly include the Round-3 consensus wording (*"incremental-with-deletions: conformance-tested across 1,000 deletion/cycle sequences, unproven"*, matching `Ek5RefCli.lean:30-31`).
   - `formal/conformance_report.json:216-236` cleanly records `153` non-semantic diffs (`pos_cyc: 104`, `reason: 49`, `semantic_fields: 0`) between `inc` and `fresh`/`lean`, caused by `ek5_v3_stratified.py:665`'s Pitfall-B shield pruning (nodes already `val=F` with `sigma > 0` skip dirty-cone re-entry when a later positive cycle is added, leaving `pos_cyc=False` / `reason="REFUTED"` while keeping `eff_auth/val/cur/eligibility` 100% exact). Document this expected EM-3 certificate-pruning behavior directly in `conformance_report.json` / `Statement.lean` so external auditors don't mistake `total: 153` for a soundness defect.
4. **`MINOR-F4` — `oracle_vals` Reads `store.refs` Instead of Re-Filtering Raw Edges (`formal/run_conformance.py:339`):**
   - `oracle_vals` passes `store.refs` (already authority-filtered by Stratum 1 of `StratifiedEvidenceStore`) into `compute_alternating_wfs_oracle`. Mitigated because `ek5_ref_cli` independently computes Stratum 1 from raw `snapshot["edges"]` and achieves `0/7,722` divergences vs `fresh` on `eff_auth`, `blocked`, and `val`. Filtering `snapshot["edges"]` inside `oracle_vals` directly would make the Python oracle 100% independent of `store.refs`.
5. **`MINOR-F5` — Wave-15 Empirical Telemetry Hooks Mentioned in §3.2 (`spherical_migration_bridge.py:238`, `formal/run_conformance.py`):**
   - `§3.2` notes logging empirical Procrustes residual beside `captured_energy_ratio` (`Theorem 1b`) and `% BEIR queries clearing Δ_cert` (`Theorem 2`). Neither is wired yet inside `/formal` (acceptable for pure Wave-14A proof bundle; track for Wave 15 / `v7.0` closeout).
6. **`MINOR-F6` — `nanoda` Pretty-Printer Warning & Staleness Check (`formal/run_conformance.py:768, 782`, `scripts/verify_all_offline.py:1079`):**
   - `nanoda_bin` emits `1 pretty printer errors: ["Unable to print axioms"]` (purely a display-printer quirk; typechecker errors = `0` and `unpermitted_axiom_hard_error: true` still enforced). Still worth documenting and asserting `0` typechecker errors explicitly in `verify_all_offline.py` alongside `rc == 0`.

---

## Part 3: Mandatory Pre-Launch Fix Checklist

1. **Before running Wave-14B Live GPUs (`Spark #1 / #2 / #3`):**
   - Fix **`BLOCKER-1`** (`Tier2ACanonicalScanner` 10-char vs 10-word n-gram index + `R5` FPR test gate).
   - Fix **`BLOCKER-2`** (`RestrictedVaultIndex._reference_rq` precomputed `ref_mean`-centered `_ref_unit_c` matching `build_vault_sidecars` + `(n, d)` `doc_mean` safety).
   - Fix **`BLOCKER-3`** (`screen_prompt` `{alias: work_id}` contract alignment).
   - Fix **`BLOCKER-4`** (`ClosedLoopStreamGate` `Profile B` full exposed-token accounting + `h0` vs `lcs_min_tokens` holdback alignment).
   - Fix **`BLOCKER-5`** (real public-domain passages for Phase 0, per-generator confirmation, real LLM execution for `Arm 1`/`Arm 2`/`Arm 7` and `TEST` streams, per-generator `nine_arm_matrix`).
   - Fix **`BLOCKER-6`** (native `MiniLM-384` unlabeled query means for Phase-3 Arm (a)).
   - Fix **`BLOCKER-7`** (`threading.Lock()` + `pad_token` guard in `run_spark3_embed_server`).
2. **Before Sealing `v7.0`:**
   - Fix **`BLOCKER-8`** (portable `formal/nanoda_manifest.json` with pinned `lean4export`/`nanoda_bin` commits, `NANODA_BIN`/PATH lookup, and `NOT_RUN` fallback).
   - Address `MINOR-1`–`MINOR-4` and `MINOR-F2`–`MINOR-F3` for clean external auditability.
