# FE panel convergence — Alex Rivera

## Position

The frontend is an operator decision surface over mutable research artifacts, not a passive skin. The panel converges on six defects: four High and two Medium. The dominant pattern is plausible presentation without browser-path proof: handlers can be unit-correct while the real page is unreachable, errors look like valid zero data, stale artifacts look live, and an obsolete report format looks supported. This is L4 with a strong L5 boundary. No FE evidence supports L14, L15, or L16 as observed violations; do not add tags to fill the taxonomy.

## Evidence lenses

1. **Can an operator reach the UI under every supported auth mode?** `dashboard_server.py:1541-1543,1591-1601` requires a Bearer header for the page and APIs; `dashboard/index.html:458-520` supplies none. A real token HTTP probe returned page 401 without header and 200 with it. Curl reachability is not browser usability. Prefer an explicit session/bootstrap contract; never embed the environment token.
2. **Does UI distinguish transport failure from empty data?** Several loaders bypass the existing status-aware helper and coerce 401/404/500 envelopes into zeros/empty arrays. Preserve last-good data only with stale/error markers; silent fallback is an operator lie.
3. **Is every displayed artifact reproducible?** The Test tab names nonexistent `generate_report_json.py`, consumes a pytest schema, and conflicts with canonical unittest guidance. Old `.report.json` presence is not a producer contract.
4. **Can artifact data become executable DOM?** Multiple fields flow to `innerHTML` in both external and inline clients. Use DOM nodes and `textContent`; producer validation alone cannot make persistent artifact text trusted.
5. **Do lifecycle words come from durable state?** Any sweep rows become “Running,” while the producer leaves the same JSON after completion. Backend must own an atomic lifecycle manifest; frontend must label legacy state unknown.
6. **Can keyboard users reach primary functions?** Five click-only `div` tabs have no focus or tab semantics. This blocks four of five views and is functional, not cosmetic.
7. **Are two dashboard clients one contract?** The static client and inline fallback duplicate rendering and sinks. Enumerate both in every AC or collapse to a small degraded fallback; avoid a wholesale rewrite presented as a narrow edit (L16 risk).
8. **What does release smoke prove?** `scripts/smoke_pipeline.py` exercises AntigravityEngine, not dashboard JS/auth/rendering. Sixty-seven unit tests passed while live HTTP disproved auth usability: the exact L5 boundary.
9. **Are zero, missing, invalid, and stale first-class states?** Research evidence requires discriminated states, schema version, source SHA/config, and generated-at/lifecycle metadata. Rejecting legacy ambiguity is safer than laundering it as current.

## Strengths to preserve

- Non-loopback binding fails closed without a token (`dashboard_server.py:2057-2060`).
- Bearer comparison uses `hmac.compare_digest`; token mode does not emit wildcard CORS.
- Newer tables already use `createElement`/`textContent`, proving the safe pattern exists.
- `fetchJson`, `asObject`, `asArray`, and `data_status=not_generated` are sound primitives when used consistently.
- Focused unit tests are valuable regression coverage beneath, not instead of, a browser smoke.

## Evidence boundary

Proven: exact source; all active PR FE diffs/comments; absence of the named producer; static DOM; 67 focused tests; real token HTTP statuses. Not proven: browser login/session because none exists; real-browser hostile-payload execution; screen-reader behavior; remote production deployment; currentness of research artifacts. These remain acceptance work, not assumed passes or failures.
