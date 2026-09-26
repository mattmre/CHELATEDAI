# Dashboard HTTP contract v1

schema: dashboard-http-v1
handler_sha: 828bd194dad06fc8ac6ec111de7a1d26e7f700d6
stacked_on: pull request 340 (`fix/track-r-residuals-20260926`)
applies_to_origin_main_2bac5d3: no

This file is the versioned handler contract for that SHA. It is not
`test_api_contract.py`. It does not describe `origin/main`
`2bac5d3f88941cf3d3034909b796e811daf31c9f`. On that commit, model-scope
and `/api/evidence_cleanup_plan` still keep a default for a non-integer
`limit`, and HEAD of the control page is still behind the bearer check.
Those sentences become false only when commit `828bd19` is on main.

`FINDINGS.md` from 2026-09-02 and `joint_fe_be_limit_cap_contract.py`
are not reconstructed here.

## Auth

`CHELATED_DASHBOARD_TOKEN` empty and
`CHELATED_DASHBOARD_ALLOW_UNAUTHENTICATED` unset: `_is_api_authorized`
is false (`dashboard_server.py`).

A query-string `token`, `access_token`, or `auth` value does not
authorize. The check reads only `Authorization: Bearer`.

GET `/`, `/dashboard`, and `/dashboard/` call `serve_dashboard` before
that check.

HEAD of those three paths returns 200, `Content-Type: text/html`,
`Content-Length: 0`, and an empty body, before the bearer check.

GET or HEAD `/api/*` without a bearer is 401 `{"error": "Unauthorized"}`.
POST, PUT, DELETE, and PATCH of any path without a bearer are 401.
With a bearer, those four methods are 405 `{"error": "Method not allowed"}`.

OPTIONS does not require a bearer. The status is 204 and the body length
is 0. `Access-Control-Allow-Origin` is sent only when
`DASHBOARD_CORS_ORIGIN` is set. There is no wildcard origin.

A path that is not an API route and is not under `/dashboard/` is 404.
A static path with `..` or a dot segment is 404. This document does not
claim a browser fetched `/.git/HEAD`.

## Integer limit

These routes return 400 `{"error": "limit must be an integer"}` when
`limit` is present and not an integer:

- `/api/events`
- `/api/campaign_history`
- `/api/validation_history`
- `/api/preflight_history`
- `/api/evidence_chain_history`
- `/api/model_scope/events`
- `/api/model_scope/features`
- `/api/model_scope/interventions`
- `/api/evidence_cleanup_plan`

`_limit_or_default` still returns its default for a non-integer. The
nine routes above reject before they use that helper, or they parse
with `int` and send 400 themselves. Other callers of
`_limit_or_default` are outside this list.

`keep_latest=abc` on evidence-cleanup still keeps `keep_latest` 1.
That parameter is not `limit`.

A numeric limit is capped at `_MAX_API_LIMIT` (5000). Limit 0 selects
no rows.

## Chain history

`load_evidence_chain_history` sets `summary.unreadable_reports` to the
count of `evidence_chain_summary.json` files that raise `OSError` or
`ValueError`. `passed` and `failed` count only parsed files.
`total_reports` counts every matched path. A corrupt file is not silent.

## What this SHA does not claim

The sweep still reuses one engine. A full collection is rewritten with
`embed_raw` before the baseline; that is pull 340, not a new grid.
The 7350-configuration grid was not run. A failed compensating upsert
logs `offline_distillation_mixed_store` and does not log training
complete. It does not write the old vectors back.

No browser was run for this document. Paint, duration cards, and
script-tag rendering are not this contract.
