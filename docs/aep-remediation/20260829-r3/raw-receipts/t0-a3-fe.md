# T0 A3 frontend receipt

## Commands

```powershell
git rev-parse HEAD
git branch --show-current
git status --short -- dashboard/index.html dashboard_server.py test_dashboard_server.py presentation.html
git ls-files -s -- dashboard/index.html dashboard_server.py test_dashboard_server.py presentation.html
git ls-files '*.html' '*.css' '*.js' '*.jsx' '*.mjs' '*.cjs' '*.ts' '*.tsx' package.json package-lock.json pnpm-lock.yaml yarn.lock 'playwright.config.*' 'cypress.config.*'
rg -c '^\s+def test_' test_dashboard_server.py tests/test_e2e_smoke.py
git grep -n -I -E '(/dashboard/?|serve_dashboard|get_inline_dashboard_html|HTTPServer\(|urlopen\(|playwright|selenium|webdriver|puppeteer|cypress|axe-core|pa11y)' -- test_dashboard_server.py 'tests/*.py'
```

## Exact source identity

```text
branch=codex/prime-ring-onion-method-dev
HEAD=65c9085cd048e8a7351a53e87666fdd5639e612b
FE_PATH_STATUS=CLEAN
100644 51507993a60ce07c24bd04964a4182308cdcaf31 0 dashboard/index.html
100644 f76fee3d0c021a685e3e604285624075c61ca111 0 dashboard_server.py
100644 ef246605c0eced6f2614b25cb9c0792c9660a097 0 test_dashboard_server.py
100644 c9a47382fb2a3ada3b798eb6fefdc7fd091d59ac 0 presentation.html
```

Sizes: static page 50,192 bytes/1,190 lines; server 84,235/2,112; dashboard tests 50,461/1,165; presentation 43,432/1,001. The only tracked web-extension files are the two HTML files; there is no standalone JS/CSS/TS file, frontend package manifest, or browser-runner configuration.

## Route/state inventory

```text
HTTP_GET_HANDLER_COUNT=1
EXPLICIT_JSON_API_COUNT=19
HANDLE_API_METHOD_COUNT=19
PAGE_ALIAS_COUNT=3
STATIC_CLIENT_API_REFERENCES=12
INLINE_CLIENT_API_REFERENCES=9
CLIENT_API_UNION=19
CLIENT_API_OVERLAP=/api/events,/api/summary
STATIC_TAB_COUNT=5
PRESENTATION_ANCHOR_SLIDE_COUNT=10
DASHBOARD_TEST_METHOD_COUNT=67
E2E_SMOKE_TEST_METHOD_COUNT=2
CHECKED_IN_SERVED_PAGE_BROWSER_HARNESS=NO_MATCH
TRACKED_A11Y_KEYBOARD_TEST_PATTERN=NO_MATCH
```

The nineteen API paths are events, summary, sweep/test/BEIR results, campaign/validation/preflight history, evidence index/chain/cleanup, Phase C results/analysis, Model Scope events/features/interventions, TTS status/last result, and disk-LLM estimate. No test starts the real dashboard server or drives the served page.
