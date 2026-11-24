# Wave 10 Factory Memo — D8 Compliance Copilot Architecture & UX (`L3`)

- **`panel_type`:** `architecture-planning`
- **`artifact`:** `docs/research/council-logs/wave10-factory-sonnet55-high.md`
- **`run_date`:** `2026-11-23`
- **`run_id`:** `wave10-r0-d6-d10-spec` / `L3`
- **`challenge_depth`:** `1`
- **`active_experts`:** `["claude-sonnet-5-5 (Factory, High)"]`
- **`persist_version`:** `1.1.0`
- **Disclosure:** `PANEL_OPERATING_INSTRUCTIONS.md` and `verify_non_nda_ip_clean.py` are absent from this tree. Header fields come from the master spec, and the `55` patterns are unverified.

## 1. EK5 v3 edge semantics (read from code)

| Intent | Edge `src -> dst` | Effect |
|---|---|---|
| Erase / retract | `REFUTE` retractor -> target | Target `val=F`, cascading through every `D_V` child (`COPY_OF`, `DERIVES_CURRENT`, `AMENDS`, `DERIVES_SNAPSHOT`) -> `AUDIT_ONLY` |
| Restate | `SUPERSEDE` newer -> older | Older `cur=F`, cascading only through `D_C` (`COPY_OF`, `DERIVES_CURRENT`). `DERIVES_SNAPSHOT` children survive |
| Audit memo | `CITES_AUDIT` memo -> cited | No stratum effect, so the memo stays `ADMIT` |

- **Spec defect:** `DERIVES_VALUE` is not in `V3_VALID_EDGE_TYPES`, and `apply_mutation` raises `ValueError`. Use `DERIVES_CURRENT`.
- **GDPR Art. 17:** use `REFUTE`, not `SUPERSEDE`, so CRM summaries flip to `AUDIT_ONLY`.
- **SEC 8-K:** non-reliance uses `REFUTE(8-K -> 10-Q)`, and echoes flip through `COPY_OF`.
- **Peer deadlock:** mutual `REFUTE` between equal-tier auditors gives `U`/`CONTESTED`. A higher-tier `REFUTE` resolves it, which makes a good "ruling" button.
- **Traps:** keep `domain="general"` on every fixture, since a mismatched domain gives authority 0 and silently kills the refute. A `REFUTE` also needs `eff_auth(src) >= eff_auth(dst)`, and failures land in `blocked_authority_edges`.

## 2. Defects in the current `server.py` and kernel usage

1. `CONTESTED_ABSTAIN` is store-global. Use slot-scoped `resolve_slot`, or a dispute in W1 abstains W2.
3. `kernel_exec_ms` times a full store rebuild. For the `<10 ms` claim, report `work_stats["wall_ms"]` on a warm store as p50/p95 over N repeats, and scope it to fixture size.
4. Responses include `text` for `AUDIT_ONLY` nodes, which leaks erased PII. Redact it. Keep payloads in a vault with a SHA-256 tombstone, and put hashes only in receipts. AUDIT_ONLY is not erasure.
5. `as_of` drops inactive attestations at every time, so a withdraw rewrites history. "Undo" must append a compensating event.
6. `CITES_AUDIT` alone lets a memo quoting a retracted dose reach clinicians. Add a Copilot-layer audience filter that shows non-`ADMIT` records and their citers only to `safety_investigator`, with a `[WITHDRAWN]` banner. This is policy outside EK5, so label it that way.
7. Add an `erased_content_sha256` blocklist so re-ingestion auto-`REFUTE`s.
8. The dispatcher drops query strings and headers, and CORS is `*`. Mount `/copilot` same-origin, and return 422 for invalid edges and `SupersedeCycleError`.

## 3. Three-arm live comparison

- All arms share one corpus, real `FastBM25Index`, dense cosine, and RRF `k=60`. Dense is MiniLM if importable, else hashed n-grams, and the receipt records which.
- **Arm 1:** Hybrid RAG with a real delete-by-id (steelman), so any leak shown is lineage-blindness only.
- **Arm 2:** `(entity, slot)` last-write-wins, labelled "Mem0-style policy, NOT live Mem0". It fails on deadlock and lineage.
- **Arm 3:** `StratifiedEvidenceStore` plus `compute_alternating_wfs_oracle`.
- Ground truth is the scenario-authored `expected_admit`. Custom injection reports oracle agreement only, with no accuracy claim, and is capped at 500 events, 2,000 edges, and 64 KB.

## 4. NDA vs Non-NDA

- **Allowlist, not scrub:** build responses from a typed public projection. A fail-closed regex pass is the second belt.
- **Label map:** keep it server-side, since a client-side map ships the forbidden terms as keys.
- **Gating:** default to `non_nda`. `?mode=nda` needs a server-verified token, because a bare query flag is investor-flippable. `--public` 404s `/` and `/api/v1/*`, which expose internal fields such as `wfs_stratum2_val`.
- **Leak surfaces:** keys, headers, errors, receipts, filenames, and user text (a reserved term gets a 422).
- **Tests:** scan every scenario x step x role response plus HTML and JS. Include a seeded-leak positive control, and assert `len(patterns)==55` from a shared patterns file.

## 5. Receipt

- Canonical JSON with integer microseconds only, since floats break canonicalization.
- Ledger chain `h_i = SHA256(h_{i-1} || canon(entry_i))`.
- Ed25519 if `cryptography` is importable, else HMAC labelled "shared-secret, not third-party verifiable".
- Disclose the ephemeral key and the untrusted server-clock timestamp.
- Include the kernel file SHA-256, the oracle mismatch count, and `transitioned` IDs as the revoked lineage.
- Name it "Retraction Audit Receipt", with a not-legal-advice line.
