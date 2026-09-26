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
`CHELATED_DASHBOARD_ALLOW_UNAUTHENTICATED` not one of `1`, `true`,
`yes`, or `on` (compared after strip and lower): `_is_api_authorized`
is false (`dashboard_server.py`). The values `0`, `false`, and empty
stay unauthorized.

A query-string `token`, `access_token`, or `auth` value does not
authorize. The check reads only `Authorization: Bearer`.

GET `/`, `/dashboard`, and `/dashboard/` call `serve_dashboard` before
that check.

HEAD of those three paths returns 200,
`Content-Type: text/html; charset=utf-8`, `Content-Length: 0`, and an
empty body, before the bearer check.

When `_is_api_authorized` is false, every other GET or HEAD path is 401
`{"error": "Unauthorized"}` before the static check. That includes
`/nope`, `/dashboard/../secret`, and `/api/*`. A missing event log is
not consulted. `GET /api/summary` is 401 in that state, not the
handler's file-not-found 404.

When the token is empty and that variable is one of `1`, `true`,
`yes`, or `on`, authorization is true and no bearer is compared.
HEAD `/api/*` is then 405 `{"error": "Method not allowed"}`.
GET `/api/summary` reaches `handle_api_summary`. A missing log file
there is 404 `{"error": "Not found"}`. GET of a path that is not an
API route and is not an allowed `/dashboard/` asset is 404 only after
authorization. A `..` or dot-segment is rejected by
`_is_static_path_allowed` at that point, also as 404.
POST, PUT, DELETE, and PATCH are 405 in this open mode with no
`Authorization` header and also with `Bearer x`.

When a token is configured, those four methods are 401 until the bearer matches. A non-matching bearer is 401. They are 405 only after the bearer matches. An `Authorization` header does not authorize by itself while the token is empty and open mode is off.

OPTIONS does not require a bearer. The status is 204 and the body length
is 0. `Access-Control-Allow-Origin` is sent only when
`DASHBOARD_CORS_ORIGIN` is set. There is no wildcard origin.

This document does not claim a browser fetched `/.git/HEAD`.

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
