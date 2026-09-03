# AEP-20260902-AUTH-01 — Fix record (P1, closeout)

- id: AEP-20260902-AUTH-01 | layer: L9 | severity: High
- file: `dashboard_server.py:1541` (gate), `:1597-1608` (`_is_api_authorized`); FE leg `dashboard/index.html`
- disposition: CONFIRMED (carried from `docs/aep-remediation/20260902-closeout/00-disposition-log.md`; challenger Sofia A3 Tier-B DISPROVE-FAILED). No re-scoring here — fix agent does not score own work.
- theme: fail-closed-auth
- branch: `aep/high/AEP-20260902-AUTH-01-L9-fail-closed-auth` (sanitized: `|`/`:` are refspec-hostile, mapped to `-`)

## Root cause (carried)

Gate was `if DASHBOARD_TOKEN and not self._is_api_authorized(): 401` with
`_is_api_authorized` returning True when no token is set: open-by-default.
FE sent zero `Authorization` headers (6 raw `fetch(` sites in
`dashboard/index.html`, ~8 in the inline fallback HTML).

## Fix (minimal, BE + FE + doc)

BE (`dashboard_server.py`):
- New `DASHBOARD_ALLOW_UNAUTHENTICATED` from `CHELATED_DASHBOARD_ALLOW_UNAUTHENTICATED`
  (1/true/yes/on); default off.
- `_is_api_authorized`: no token → returns the flag (fail-closed default);
  token set → unchanged `Bearer` + `hmac.compare_digest` check.
- `do_GET` gate: `if not self._is_api_authorized(): 401` (covers all GET routes).
- `run_server` startup line now prints effective mode
  (`enabled` | `fail-closed (no token)` | `open (explicit ...)`).
- Rotation documented in code comment: replace `CHELATED_DASHBOARD_TOKEN`, restart.

FE (`dashboard/index.html` + inline fallback in `dashboard_server.py`):
- One `window.fetch` wrapper per surface tagging relative `/api/*` requests with
  `Authorization: Bearer <token>`; token from `?token=` (persisted to
  sessionStorage) → sessionStorage → localStorage. No call-site changes; no
  header on non-`/api/` URLs so nothing leaks to third parties (CDN uses
  `<script src>`, untouched — CDN-01 out of scope).

Tests (`test_dashboard_server.py`):
- 3 routing `do_GET` tests declare `_explicit_open_mode()` (routing intent
  preserved under new default); import block normalizes flag to False.
- `TestDashboardSecurity` setUp/tearDown manages the flag; +2 AC3 tests
  (fail-closed 401 default; routed in explicit open mode).

## AC verification (live loopback, port 8931, seeded `--log-file`)

- AC1 token-on, no header → 401: PASS (got 401)
- AC2 token-on, valid Bearer → 200: PASS (got 200); wrong Bearer → 401: PASS
- AC3 token unset, default → 401 fail-closed: PASS (got 401)
- AC3 token unset + `CHELATED_DASHBOARD_ALLOW_UNAUTHENTICATED=1` → 200: PASS
- Note: first probe round 404'd the 200-cases because `chelation_events.jsonl`
  is absent in-tree (pre-existing `FileNotFoundError → 404` handler path,
  unrelated to auth); re-ran with seeded `--log-file`, all PASS.
- FE served check: `GET /dashboard` body contains shim (4 token markers,
  1 Bearer assignment, 1 `/api/` guard); extracted shim passes `node --check`.
- Unit: `pytest test_dashboard_server.py` → 67 passed, 2 skipped (both skips
  pre-existing: `.report.json` / `benchmark_beir_results.json` present in cwd).
- `py_compile` clean. ruff unavailable in env (not installed) — not run.

## Non-goals / residuals

- CORS `*` fallback untouched (CORS-01 ordered after AUTH-01, separate finding).
- STATIC-01 untouched; `do_HEAD` static path noted as its territory, not widened here.
- BHS scoring explicitly NOT done by fix agent (needs fresh Tier-B agent).
