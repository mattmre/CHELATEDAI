# Panel of Experts Review: Infrastructure & Cloud
**ChelatedAI Repository — Session 32**
**Date:** 2026-04-04
**Panel:** Lisa Park (Cloud Architect), Omar Hassan (DevSecOps), Thomas Eriksen (Platform Reliability), Nina Chen (Python Packaging), Victor Reyes (Firmware Build), Devil's Advocate

---

## Panel Mandate

Examine every infrastructure, packaging, CI, dependency, and environment quality issue in the ChelatedAI repository. Files reviewed:

- `.github/workflows/test.yml`
- `.github/workflows/build_firmware.yml`
- `pyproject.toml`
- `requirements.txt`
- `computational_storage_poc/firmware/` (all files)
- `computational_storage_poc/emulation/` (all files)
- `computational_storage_poc/validation_config.py`
- `CLAUDE.md`
- `docs/computational-storage-retention-policy-2026-03-06.md`
- `.gitignore`
- `SECURITY.md`
- `CODEOWNERS`

---

## CONVENE — Panel Charter

The panel is chartered to find every issue across six lenses: CI/CD architecture, environment parity, dependency management, build reproducibility, packaging best practices, supply chain security, secret handling, build reliability, flaky test surface, CI efficiency, cache invalidation, pyproject.toml completeness, firmware build pipeline completeness, artifact retention, cross-platform compatibility, linting configuration, and release automation gaps. No issue too small to surface; the project owner decides what to act on.

---

## SOLO REVIEWS

---

### Lisa Park — Cloud Architect (16 years)

**Lens:** CI/CD architecture, environment parity, dependency management, build reproducibility, packaging best practices.

**LP-01 [HIGH] torch not in requirements.txt creates pip cache miss and double-install**
The `test` job installs PyTorch explicitly (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) and then installs `requirements.txt` which also specifies `torch>=2.0`. This results in torch being resolved twice — once from the PyTorch CDN index and once from PyPI's index in requirements. pip may re-download or silently upgrade torch on the second pass. More importantly, `actions/setup-python` with `cache: "pip"` keys the cache off `requirements.txt` content. Because torch is not in `requirements.txt`, changes to the separate install step never bust the cache. If the torch install step is modified, CI will silently continue using old cached artifacts.

**LP-02 [HIGH] No job dependency ordering — parallel jobs can mask failures**
`test.yml` has four jobs (`lint`, `test`, `computational-storage-fundamentals`, `computational-storage-emulation`) that run in full parallelism with no `needs:` declarations. There is no requirement that lint passes before tests run. A PR with ruff violations will still trigger full test matrix execution, burning CI minutes even though it will eventually fail on lint. Additionally, there is no check that `computational-storage-fundamentals` passes before `computational-storage-emulation` runs, even though the emulation job depends on the same emulation stack.

**LP-03 [HIGH] test matrix job runs all 48 test files including computational-storage tests (duplicate execution)**
The main `test` job runs `python -m unittest discover -s . -p "test_*.py" -v` which discovers all 48 `test_*.py` files, including `test_computational_storage_poc.py`, `test_computational_storage_emulation.py`, `test_computational_storage_hardware_evidence.py`, and `test_computational_storage_payload.py`. These same files are re-executed by the `computational-storage-fundamentals` and `computational-storage-emulation` jobs. Every PR runs computational storage tests 3–5 times (4 Python versions in matrix + 2 dedicated jobs). This wastes CI minutes and creates no additional signal.

**LP-04 [MEDIUM] No environment parity between development (Windows) and CI (ubuntu-latest)**
CLAUDE.md documents that the developer runs on Windows 11 (`win32`). All CI jobs run `ubuntu-latest`. The `test_computational_storage_hardware_evidence.py` file contains `\\.\PhysicalDrive2` path tests (Windows raw device paths). These tests are written to mock the Windows-specific behavior, but as CLAUDE.md notes, "Explicit Windows raw-device paths like `\\.\PhysicalDrive2` are valid inputs." There is no Windows CI runner verifying Windows-specific code paths. A breakage on the primary development platform would not be caught by CI.

**LP-05 [MEDIUM] Firmware build clones pico-sdk on every CI run (no caching)**
The `build_firmware.yml` workflow clones `https://github.com/raspberrypi/pico-sdk.git --branch master --depth 1` and then calls `git submodule update --init` on every run. The Pico SDK is approximately 200MB. There is no `actions/cache` step for the SDK clone. Every firmware build from scratch downloads 200MB of SDK. Over a year of active development with weekly firmware changes, this accumulates into tens of GB of external bandwidth and 3–5 minutes of clone time per run.

**LP-06 [MEDIUM] No container-based environment isolation for the main Python test jobs**
The test matrix runs directly on the `ubuntu-latest` hosted runner environment. Packages from previous runs or from GitHub's pre-installed software can interfere. In particular, `ubuntu-latest` pre-installs Python packages that may shadow or conflict with `pip install`. For a research prototype this is acceptable, but it means environment parity with a clean install is not guaranteed.

**LP-07 [MEDIUM] No release workflow — version is hardcoded static `0.1.0`**
`pyproject.toml` has `version = "0.1.0"`. There is no release GitHub Actions workflow, no version bump automation, and no git tag convention. If the project ever needs to distribute a release, there is no defined process. The version number has not changed since initial setup despite significant feature additions.

**LP-08 [MEDIUM] `computational-storage-fundamentals` job runs `run_all_tests.py` which uses subprocess**
The CI step `python computational_storage_poc/run_all_tests.py` is a script harness that invokes other scripts via `subprocess.run`. This bypasses the standard test runner and produces no JUnit/TAP/structured output. If one of the sub-scripts fails, the error output is interleaved with harness output and not parseable by GitHub's test result rendering. No test counts are reported.

**LP-09 [LOW] `ubuntu-latest` is a moving target**
Both workflows use `runs-on: ubuntu-latest` without pinning to a specific Ubuntu version (e.g., `ubuntu-22.04`). When GitHub updates `ubuntu-latest` (e.g., from Ubuntu 22 to 24), the runner environment changes — different default Python version, different ARM GCC version in apt, different system library versions. This can cause silent regressions in the firmware build specifically.

**LP-10 [LOW] No workflow concurrency controls**
Neither workflow has a `concurrency:` block. If a developer pushes multiple commits quickly to a PR branch, all pushes will queue CI runs simultaneously. This is wasteful on public runners and could cause unexpected interactions in a self-hosted environment. GitHub recommends `concurrency: { group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true }` for typical workflows.

**LP-11 [LOW] pip install ruff runs without `--upgrade` and may use cached stale version**
The lint job runs `pip install ruff>=0.4` but `actions/setup-python` with `cache: "pip"` might serve a cached ruff that satisfies `>=0.4` but is months behind the current release. Ruff has weekly releases; the pinned minimum of `0.4` was set months ago and ruff is now at 0.9+. New lint rules added in ruff 0.5–0.9 are never applied.

**LP-12 [LOW] No CI step to validate pyproject.toml integrity**
There is no CI step that runs `pip check` or `python -m build --check` to verify that `pyproject.toml` is self-consistent and that all declared `py-modules` exist on disk. If a developer adds a module and forgets to add it to `py-modules`, CI will not catch it.

**LP-13 [LOW] No documentation build pipeline**
There is no CI job that validates the documentation. The `docs/` directory contains 50+ markdown files. No link checker, no markdown linter, no doc build step. Broken internal links (`docs/INDEX.md` references files that may not exist) are never detected.

**LP-14 [LOW] No artifact signing or verification for firmware UF2**
The firmware artifact (`computational_storage_firmware`) uploaded by `build_firmware.yml` is unsigned. Anyone with repository access can download and flash this UF2. For a POC this is acceptable, but there is no attestation that the artifact was built from a specific commit.

**LP-15 [LOW] Emulation Docker image is not published to a registry**
`computational_storage_poc/emulation/Dockerfile` and `docker-compose.yml` define a Docker image but it is never built and published as part of CI. Every local user must `docker compose up --build` to get the image. There is no pinned base image tag (`python:3.10-slim` is floating).

---

### Omar Hassan — DevSecOps Engineer (12 years)

**Lens:** Supply chain security, dependency vulnerabilities, secret handling, CI security, access controls.

**OH-01 [CRITICAL] No dependency lock file — supply chain is completely unpinned**
`requirements.txt` uses minimum version bounds only (`numpy>=1.24`, `torch>=2.0`, `sentence-transformers>=2.2`, `qdrant-client>=1.7`, `requests>=2.28`, `mteb>=1.0`, `scikit-learn>=1.5`). There is no `pip freeze` lockfile, no `poetry.lock`, no `uv.lock`, no `pip-compile` output. Every CI run can silently resolve different transitive dependency versions. This is a textbook supply chain attack surface: a malicious release of any transitive dependency will be automatically picked up. The `mteb>=1.0` constraint alone has 15+ transitive dependencies that are completely unconstrained.

**OH-02 [CRITICAL] GitHub Actions use floating action versions with no SHA pinning**
Both workflows pin actions by semver tag (`actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`) rather than by commit SHA. A tag like `v4` is a mutable reference — the tag can be force-pushed to point to a different commit. A compromised `actions/checkout@v4` would execute arbitrary code with write permissions to the repository. Best practice (required by most enterprise security policies) is to pin to the full commit SHA: `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4`.

**OH-03 [HIGH] No explicit `permissions:` block in workflows — default permissions are over-broad**
Neither `test.yml` nor `build_firmware.yml` declares a `permissions:` block. GitHub Actions defaults to repository read/write permissions when no block is set (depending on repository settings). The principle of least privilege requires that CI workflows declare exactly what they need: `permissions: { contents: read }` for pure test/lint workflows, nothing else. Without this, a compromised step can push commits, create releases, or modify branch protection.

**OH-04 [HIGH] No dependency vulnerability scanning (Dependabot or similar)**
There is no `dependabot.yml` in `.github/`. Dependabot is not enabled for Python dependencies, GitHub Actions versions, or Docker images. Known CVEs in `torch`, `sentence-transformers`, `numpy`, or `qdrant-client` will not generate automated PRs. The project has no defined process for responding to vulnerability disclosures in dependencies.

**OH-05 [HIGH] `fusepy` is an unmaintained package installed in the emulation Docker image**
`Dockerfile` runs `pip install fusepy numpy`. `fusepy` (`fuse-python` / `fusepy`) has had no releases since 2016 and the primary fork is abandoned. It has known security issues related to privilege escalation through FUSE mount operations. The Docker container runs with `privileged: true` (see docker-compose.yml), compounding the risk. Although this is a local development tool, the combination of privileged container + abandoned FUSE library is a significant attack surface if the emulator is ever run in a shared environment.

**OH-06 [HIGH] `adapter_weights.benchmark-backup-*.pt` files are untracked but present locally — no gitignore coverage**
Three large binary files are in the working tree at the repository root:
- `adapter_weights.benchmark-backup-808fc792cae04d3da85b5d394b3e26a2.pt` (2.3MB)
- `adapter_weights.benchmark-backup-f658c34e70e047e0a0faaab68c7b5837.pt` (0.6MB)
- `adapter_weights.benchmark-backup-fa0a54d79ce1477191b68ac54114458b.pt` (2.3MB)

These are not gitignored (confirmed via `git check-ignore`). They appear only in `.git/info/exclude`. This is a local-only gitignore — it is not tracked and does not protect collaborators or CI from accidentally committing these files. A `git add -A` or IDE auto-commit could include multi-megabyte binary model weights in the repository history. The pattern `adapter_weights.benchmark-backup-*.pt` should be added to `.gitignore`.

**OH-07 [MEDIUM] `sweep_results.json` has no gitignore entry**
`sweep_results.json` is visible in the working tree and is NOT covered by any `.gitignore` rule (confirmed). This file likely contains experiment output data. It could be accidentally committed, and in a non-research context could contain sensitive data (API keys embedded in sweep configs, internal dataset paths, etc.).

**OH-08 [MEDIUM] `nul` file is committed to the repository (tracked by git)**
A file named `nul` exists at the repository root and is tracked in git (`git ls-files` confirms). On Windows, `nul` is a reserved device name (equivalent to `/dev/null`). The file contains 46 bytes (appears to be a Windows shell redirect artifact from a `del nul` or `> nul` command). This file: (1) has undefined behavior when checked out on Windows (cannot create a file named `nul` in some Windows contexts), (2) clutters the repository, (3) was likely created by accident. It should be removed from git history.

**OH-09 [MEDIUM] Docker image uses `privileged: true` with no documented justification for CI**
`docker-compose.yml` declares `privileged: true` for the emulator service. This gives the container full access to the host kernel — capabilities equivalent to root on the host. The `build_firmware.yml` CI job does not use Docker, but if this docker-compose setup were ever used in a CI context (e.g., a future self-hosted runner), it would expose the runner to container escapes. The `privileged: true` should be replaced with targeted capabilities (`--cap-add SYS_ADMIN --device /dev/fuse`).

**OH-10 [MEDIUM] `pico-sdk` cloned from `--branch master` in CI — a mutable HEAD**
The firmware build clones `https://github.com/raspberrypi/pico-sdk.git --branch master --depth 1`. The `master` branch head is mutable. If the Raspberry Pi Foundation pushes a breaking change or (hypothetically) a compromised commit to `pico-sdk:master`, the next firmware CI run will pick it up without any review. The clone should be pinned to a specific release tag (e.g., `--branch 2.1.0`) or a specific commit SHA.

**OH-11 [LOW] No SBOM (Software Bill of Materials) generation**
There is no step in any workflow that generates an SBOM (e.g., via `cyclonedx-py` or `pip-audit`). For a project with firmware artifacts, an SBOM is increasingly expected by security auditors and is required for SLSA compliance.

**OH-12 [LOW] `SECURITY.md` links to `main` branch only — no supported version matrix**
`SECURITY.md` states "Only the latest version on the `main` branch is actively supported." There is no version number referenced, no CVE response timeline (beyond "48 hours for acknowledgment"), and no security contact other than GitHub's private advisory mechanism. For a project with firmware artifacts, this is minimal.

**OH-13 [LOW] No secret scanning configuration**
There is no GitHub Advanced Security secret scanning configuration. While the repository does not appear to contain secrets currently, there is no automated guard preventing accidental future commits of API keys, tokens, or credentials. GitHub's free secret scanning for public repositories should be explicitly enabled and verified.

**OH-14 [LOW] Requirements include `requests>=2.28` as a direct dependency but it is only used for Ollama**
`requirements.txt` lists `requests>=2.28` as a top-level dependency. `CLAUDE.md` describes it as optional ("Optional: Ollama for Docker-based embeddings"). It should be in the `[project.optional-dependencies]` `ollama` group in `pyproject.toml`, not in the mandatory `requirements.txt`. Every CI run installs `requests` even when Ollama is not used.

**OH-15 [LOW] No code scanning (SAST) configured**
There is no CodeQL, Semgrep, or Bandit step in any CI workflow. Static analysis for security vulnerabilities (injection, unsafe deserialization, etc.) is entirely absent. The `sedimentation_trainer.py`, `checkpoint_manager.py` (which uses SHA256 for verification), and the firmware C code are not scanned.

---

### Thomas Eriksen — Platform Reliability Engineer (14 years)

**Lens:** Build reliability, flaky tests, CI efficiency, cache invalidation, matrix testing.

**TE-01 [HIGH] Pip cache key does not include the separately-installed torch**
`actions/setup-python` with `cache: "pip"` generates its cache key by hashing `requirements.txt` (or `pyproject.toml` if present). Since torch is installed via a separate step (`pip install torch --index-url ...`) before `pip install -r requirements.txt`, the torch installation is effectively uncached — it must be downloaded fresh on every run. PyTorch CPU wheel for Python 3.11 is approximately 180MB. With 4 Python versions in the matrix, that is 720MB of torch download per CI run. At ~10 CI runs per week, that is 7GB/week of redundant downloads.

**TE-02 [HIGH] Test discovery runs in root `.` directory including emulation/firmware submodules**
The main test job uses `python -m unittest discover -s . -p "test_*.py" -v`. Discovery starts at the repository root (`.`). While `computational_storage_poc/` test files do not match `test_*.py` pattern, the four `test_computational_storage_*.py` files at the repo root are discovered and run by the main matrix job. These tests also import from `computational_storage_poc/emulation/` which manipulates `sys.path`. This path manipulation is global and can affect subsequent test module imports within the same discovery run. Any `sys.path` pollution introduced by emulation tests could cause flaky behavior in unrelated test files discovered later.

**TE-03 [HIGH] No test timeout configured — runaway tests can consume full 6-hour CI limit**
Neither `test.yml` nor any test file configures a test timeout. The `run_all_tests.py` harness has no timeout. If a test hangs (e.g., `sentence-transformers` downloading a model, an Qdrant operation blocking, or a subprocess deadlock), the CI job will run for up to 6 hours before GitHub cancels it. The `embedding_backend.py` with Ollama HTTP calls is particularly susceptible if the mock is incomplete.

**TE-04 [MEDIUM] Separate `computational-storage-emulation` job runs `validate_emulation_path.py` as a script, not as a test**
The step `python computational_storage_poc/emulation/validate_emulation_path.py` is invoked as a plain script, not through the unittest runner. Its output is a print statement, not a structured test result. If this validation fails, the error message is a raw `AssertionError` in the CI log. GitHub cannot parse or count this as a test failure vs. a runtime error.

**TE-05 [MEDIUM] Matrix `fail-fast: false` means all 4 Python versions always run regardless of failures**
The test matrix has `fail-fast: false`. This is intentional for cross-version compatibility testing. However, it means that a catastrophic failure (e.g., a missing import that fails all tests on all versions) will still consume 4× CI minutes before reporting failure. Consider using `fail-fast: true` with a follow-up matrix-only job triggered on failure to identify which specific version failed.

**TE-06 [MEDIUM] `computational-storage-fundamentals` job has no timeout on `run_all_tests.py`**
`run_all_tests.py` invokes six sub-scripts sequentially. `train_and_compile.py` trains a PyTorch model. There is no per-step or per-job timeout. If the training script hangs or the PyTorch download fails silently, the CI job will consume resources indefinitely.

**TE-07 [MEDIUM] ruff lint job runs on Python 3.11 only — type annotation syntax from Python 3.10+ may not be caught**
The lint job runs ruff on Python 3.11. The test matrix covers Python 3.9–3.12. ruff's `target-version = "py39"` is set correctly in `pyproject.toml`. However, the lint job does not run type checks (mypy, pyright), so Python 3.9-incompatible annotations (e.g., `X | None` instead of `Optional[X]`) are only caught if `from __future__ import annotations` is missing AND the test runner actually imports that file on Python 3.9. CLAUDE.md notes this specific concern: "If a module imported by tests uses `X | None` annotations, add `from __future__ import annotations` or use `Optional[...]`."

**TE-08 [MEDIUM] No CI job covers Python 3.13**
The matrix tests Python 3.9, 3.10, 3.11, 3.12. Python 3.13 was released in October 2024. `sentence-transformers`, `torch`, and `qdrant-client` all support Python 3.13. Not testing against it means potential regressions when users run on 3.13 go undetected.

**TE-09 [LOW] `computational-storage-emulation` job installs only `numpy` — other imports may be missing**
The step `pip install numpy` is the only dependency install in the `computational-storage-emulation` job. `test_computational_storage_emulation.py` imports `payload_contract`, `usb_host_inference`, `validate_emulation_path`, and `virtual_controller` — all in-repo modules. But if any of those modules has a transitive import of `torch`, the job will fail with an `ImportError`. Currently the emulation modules appear numpy-only, but this is fragile and not validated by the CI configuration.

**TE-10 [LOW] Firmware build has no CI caching for arm-none-eabi toolchain or apt packages**
`sudo apt-get install -y gcc-arm-none-eabi libnewlib-arm-none-eabi build-essential cmake` runs on every firmware build. `apt-get` is not cached. The ARM GCC package is large (~100MB). GitHub does not cache apt installations by default. An `actions/cache` step keyed on the package list hash would save 1–2 minutes per firmware build.

**TE-11 [LOW] No flaky test detection or retry mechanism**
There is no `--rerun-failures` or similar flaky test mitigation. The `qdrant-client` with in-memory backend and the `sentence-transformers` model loader are both known to have occasional race conditions in test environments. A single flaky failure will block a PR merge.

**TE-12 [LOW] `make -j$(nproc)` in firmware build — nproc returns the number of vCPUs, typically 2 on GitHub runners**
The firmware build uses `make -j$(nproc)`. GitHub `ubuntu-latest` runners have 4 vCPUs as of 2024. For a two-file project (`main.c`, `usb_descriptors.c`), this parallelism provides no benefit and adds minor overhead.

**TE-13 [LOW] No CI badge in README.md for build/test status**
The `README.md` does not include a GitHub Actions status badge. Contributors and visitors cannot see the current build health without navigating to the Actions tab.

**TE-14 [LOW] `computational-storage-fundamentals` job imports from `computational_storage_poc/` which manipulates `sys.path` via `os.chdir()`**
`run_all_tests.py` calls `os.chdir(os.path.dirname(os.path.abspath(__file__)))`. This changes the working directory for all subsequent subprocess calls. If any subsequent CI step in the same job relied on the original working directory, it would fail. This is safe currently since `run_all_tests.py` is the last step, but it is a fragile pattern.

**TE-15 [LOW] No CI check for `py-modules` completeness in pyproject.toml**
Seven production modules exist at the repo root that are not listed in `pyproject.toml`'s `py-modules`: `benchmark_beir`, `cross_lingual_distillation`, `isomer_detector`, `kalman_lr_scheduler`, `language_detector`, `sedimentation_loss`, `topology_analyzer`. If a user installs the package via `pip install chelatedai`, these modules will not be installed. There is no CI check that validates the `py-modules` list against the actual filesystem.

---

### Nina Chen — Python Packaging Specialist (9 years)

**Lens:** pyproject.toml correctness, requirements.txt hygiene, version pinning strategy, editable installs, distribution quality.

**NC-01 [HIGH] 7 production modules missing from `[tool.setuptools] py-modules`**
The following `.py` files exist at the project root and are imported by other modules or tests, but are NOT listed in `pyproject.toml`'s `py-modules` section:
- `benchmark_beir.py` — imported by `test_benchmark_beir.py`
- `cross_lingual_distillation.py` — listed in CLAUDE.md architecture
- `isomer_detector.py` — listed in CLAUDE.md architecture
- `kalman_lr_scheduler.py` — listed in CLAUDE.md architecture
- `language_detector.py` — listed in CLAUDE.md architecture
- `sedimentation_loss.py` — listed in CLAUDE.md architecture
- `topology_analyzer.py` — listed in CLAUDE.md architecture

A `pip install chelatedai` or `pip install -e .` will not expose these modules as installed package components. Any downstream user importing `from kalman_lr_scheduler import ...` after a pip install will get `ModuleNotFoundError`. This is a data-integrity issue for the distribution.

**NC-02 [HIGH] `requirements.txt` and `pyproject.toml` are partially diverged**
`requirements.txt` lists 7 dependencies. `pyproject.toml` `[project].dependencies` lists 5 dependencies. Divergences:
- `requests>=2.28` is in `requirements.txt` but NOT in `pyproject.toml` core deps (it is in `[project.optional-dependencies].ollama`).
- `mteb>=1.0` is in `requirements.txt` but NOT in `pyproject.toml` core deps (it is in `[project.optional-dependencies].benchmark`).

Users who install via `pip install chelatedai` will not get `requests` or `mteb`, but the `requirements.txt` implies these are mandatory. Users who use `pip install -r requirements.txt` get unnecessary packages. The two sources of truth are diverged and will continue to diverge without a merge strategy.

**NC-03 [MEDIUM] No `[project.urls]` metadata**
`pyproject.toml` has no `[project.urls]` table. A minimal distribution should include `Homepage`, `Repository`, and `Bug Tracker` URLs. Without these, `pip show chelatedai` produces no useful links and PyPI (if ever published) shows no source link.

**NC-04 [MEDIUM] No `[project.classifiers]` table**
There are no trove classifiers in `pyproject.toml`. Without classifiers, PyPI search cannot categorize the package. Minimum recommended classifiers for a research ML package: `Programming Language :: Python :: 3`, `License :: OSI Approved :: MIT License`, `Topic :: Scientific/Engineering :: Artificial Intelligence`.

**NC-05 [MEDIUM] No `[project.authors]` or `[project.maintainers]` table**
`pyproject.toml` has no `authors` field. This means `pip show chelatedai` returns no maintainer contact information. CODEOWNERS lists `@mattmre` but this is not propagated to the package metadata.

**NC-06 [MEDIUM] No `readme` field in `[project]`**
`pyproject.toml` does not declare `readme = "README.md"`. If the package were published to PyPI, the project page would have no long description. `readme` should be `readme = "README.md"`.

**NC-07 [MEDIUM] `setuptools>=68.0` in `build-system.requires` is unpinned beyond minimum**
`build-system.requires = ["setuptools>=68.0"]`. setuptools 68 was released in 2023; the current version is 75+. While setuptools is generally backward-compatible, using a floating minimum means the build backend can change behavior on any new setuptools release. For reproducible builds, this should be pinned to a tested version range (e.g., `setuptools>=68.0,<76`).

**NC-08 [MEDIUM] `ruff>=0.4` in `[project.optional-dependencies].dev` is also installed directly in CI with `pip install ruff>=0.4`**
The `dev` optional dependency group in `pyproject.toml` includes `ruff>=0.4`. The CI lint job installs ruff with a separate `pip install ruff>=0.4` step rather than using `pip install -e ".[dev]"`. This means developers and CI are not using the same installation mechanism for development tools. If the `dev` group specification is updated, CI will not pick it up.

**NC-09 [LOW] No `[tool.setuptools.package-data]` for non-Python assets**
The package has no declared package data. If `validation_config.py` constants were moved to a TOML/JSON config file, or if any future asset needed distribution, there is no `package-data` configuration. This is a forward-looking gap rather than a current bug.

**NC-10 [LOW] `py-modules` approach does not scale — should migrate to `find_namespace_packages` or explicit packages**
With 25 declared modules and 7 undeclared ones, the manual `py-modules` list is already stale. Each new module requires a manual edit to `pyproject.toml`. The existing packaging evaluation document (`docs/packaging-evaluation-2026-02-27.md`) recommends keeping the flat layout for now, but there is no mechanism to detect when the list falls out of sync. At minimum, a CI check should verify that all non-test, non-script `.py` files at the root are in `py-modules`.

**NC-11 [LOW] `langdetect` dependency is commented out in `requirements.txt` but never added to optional-dependencies**
`requirements.txt` has `# langdetect>=1.0.9` as a comment. `pyproject.toml` does not have a `langdetect` entry in any optional group. The `language_detector.py` module notes it "falls back to heuristics if absent." If a user wants langdetect support, there is no `pip install chelatedai[langdetect]` path — they must know to install it manually.

**NC-12 [LOW] No `python_requires` upper bound**
`requires-python = ">=3.9"` has no upper bound. If a future Python version (e.g., Python 3.14) introduces breaking changes, there is no mechanism to warn users. For production libraries, `python_requires = ">=3.9,<3.14"` with tested bounds is safer.

**NC-13 [LOW] `httpx>=0.27` in dev dependencies is not used in any visible test**
The `dev` optional group includes `httpx>=0.27`. A search of the codebase does not reveal any direct httpx usage in test files (the `test_dashboard_server.py` may use it for local HTTP testing). If httpx is only needed for the dashboard test, it should be noted as a dashboard testing dependency, not a general dev dependency.

**NC-14 [LOW] No `[tool.coverage]` or `[tool.pytest.ini_options]` even though coverage is referenced in `.gitignore`**
`.gitignore` includes `.coverage` and `htmlcov/`, implying coverage reporting is sometimes used. However, `pyproject.toml` has no `[tool.coverage]` section and there is no `.coveragerc`. CI has no coverage reporting step. This is an incomplete setup that creates confusion — is coverage used or not?

**NC-15 [LOW] No editable install step in any CI job**
Despite `pyproject.toml` existing and `pip install -e .` being documented in CLAUDE.md, no CI job actually installs the package. This means the `py-modules` declaration is never tested in CI. A missing module in `py-modules` would not be caught by any CI job.

---

### Victor Reyes — Firmware Build Engineer (10 years)

**Lens:** Embedded build pipelines, UF2/ELF artifact quality, firmware CI, hardware-in-the-loop testing.

**VR-01 [CRITICAL] pico-sdk cloned at `--branch master` (mutable HEAD) — reproducible builds impossible**
The `build_firmware.yml` workflow clones `pico-sdk --branch master --depth 1`. The `master` branch is a live development branch; it is NOT a stable release branch. Pico SDK has tagged releases (1.5.1, 2.0.0, 2.1.0). Using `master` means:
1. The firmware binary output can change between CI runs even with identical source code.
2. Breaking API changes on pico-sdk `master` will silently break the build.
3. The UF2 artifact cannot be reproducibly rebuilt from the commit SHA.
4. SLSA provenance (if ever required) cannot be satisfied.
The fix is `--branch 2.1.0` (or whatever the current stable tag is).

**VR-02 [HIGH] Firmware artifact name is not tied to git commit SHA or build number**
The artifact is uploaded as `computational_storage_firmware` — a static name. Every successful firmware build overwrites the previous artifact in GitHub's artifact store (after 14 days). There is no commit SHA or build number in the artifact name. If two firmware builds run close together (e.g., on a PR and a push), there is no way to determine which artifact corresponds to which commit from the artifact name alone.

**VR-03 [HIGH] No hardware-in-the-loop (HIL) test — firmware correctness is never validated against real hardware in CI**
The firmware is built but never executed. The UF2 is generated and uploaded, but no CI step reads sector 100 from a flashed Pico and validates the JSON output matches `validation_config.py` thresholds. CLAUDE.md notes: "If no RP2040 device is attached, do not fabricate hardware evidence." This is correct policy, but it means firmware logic correctness is validated only by the software emulation layer, not by the actual compiled C code running on hardware. The compiled firmware and the Python emulation could silently diverge.

**VR-04 [HIGH] No firmware version embedded in the binary**
`main.c` has no firmware version constant, no build timestamp embedding, no git commit hash embedding. The `tud_msc_inquiry_cb` returns a hardcoded product revision `"0001"` that never changes. A user who flashes the Pico and runs `usb_host_inference.py` has no way to verify which firmware version is running from the USB inquiry data alone.

**VR-05 [MEDIUM] `DISK_BLOCK_COUNT = 200` creates a 100KB virtual disk — too small for real model storage**
The firmware defines `DISK_BLOCK_COUNT = 200` (200 × 512 bytes = 100KB) for the virtual disk. The validation docs note that the firmware scope is "transport correctness and reproducible payloads, not yet full on-device parity with the trained digits model." But the 100KB limit is not documented as a known constraint in the firmware code itself. Future contributors unfamiliar with this scope decision may attempt to extend the firmware expecting more storage capacity.

**VR-06 [MEDIUM] `msc_disk` is a static RAM array — RP2040 has only 264KB of RAM total**
`uint8_t msc_disk[DISK_BLOCK_COUNT][SECTOR_SIZE]` = 200 × 512 = 102,400 bytes = 100KB of RAM. The RP2040 has 264KB total RAM. This allocation consumes 38% of available RAM before any stack, heap, or TinyUSB buffers. For the 200-block POC this works, but the implementation comment "In a real implementation, this would point directly to the RP2040's attached QSPI Flash" is critical — QSPI flash is 2MB and would be the correct storage for real use. The RAM-based approach should be flagged more prominently as a POC limitation.

**VR-07 [MEDIUM] No firmware `version.h` or CMake version injection**
There is no `version.h` header, no CMake `CONFIGURE_FILE` step to inject the git hash or semver into the firmware. A minimal embedded practice is to include the build commit SHA as a string constant that can be read back via a USB SCSI INQUIRY command or a dedicated vendor command.

**VR-08 [MEDIUM] `pico_sdk_import.cmake` is a local copy — may drift from SDK**
`pico_sdk_import.cmake` is a copy of the file from the Pico SDK itself (as noted in its header comment). If pico-sdk is updated, this local copy may diverge from the SDK's version, causing import failures. The correct approach is to reference `pico_sdk_import.cmake` directly from the cloned SDK rather than maintaining a local copy.

**VR-09 [MEDIUM] Firmware build artifact has no verification of expected output values**
The firmware build uploads a UF2 artifact but there is no CI step that disassembles or inspects the ELF to verify that the expected constant values (`kToyInput`, `kToyLayer1`, `kToyLayer2`) are embedded correctly. A compiler optimization or a linker flag change could silently alter constant folding behavior. Even a basic `arm-none-eabi-objdump -s` grep for known float values would catch gross miscompilation.

**VR-10 [LOW] `CFG_TUSB_DEBUG = 0` in tusb_config.h — debug builds are not tested**
TinyUSB debugging is disabled (`CFG_TUSB_DEBUG 0`). CI never builds with debug logging enabled. If a USB communication issue occurs, there is no CI-verified debug build available. A separate CI step that builds with `CFG_TUSB_DEBUG 1` would validate that debug builds compile.

**VR-11 [LOW] No CMake preset file (`CMakePresets.json`) — build configuration is not reproducible across environments**
There is no `CMakePresets.json` defining standard build configurations (debug, release, CI). Developers must manually run `cmake ..` and accept default flags. The CI uses the same default flags. Adding a preset for the release configuration used in CI would make local and CI builds identical.

**VR-12 [LOW] `GIT_SUBMODULES_RECURSE FALSE` in pico_sdk_import.cmake skips TinyUSB submodule**
`pico_sdk_import.cmake` calls `FetchContent_Populate` with `GIT_SUBMODULES_RECURSE FALSE`. Then the `build_firmware.yml` workflow calls `git submodule update --init` on the separately cloned pico-sdk. These two submodule init paths may not initialize the same set of submodules. TinyUSB is a git submodule of pico-sdk. If the manual `git submodule update --init` does not initialize recursively, TinyUSB headers may be missing.

**VR-13 [LOW] No `make clean` or build directory removal between CI runs**
The firmware build `mkdir build && cd build && cmake .. && make` will fail if the `build/` directory already exists from a previous run (in a non-ephemeral environment or with GitHub Actions cache). The workflow creates `build/` without checking for prior existence. On GitHub's ephemeral runners this is fine, but for self-hosted runners this is a latent bug.

**VR-14 [LOW] `README_FIRMWARE.md` documents Windows build steps — CI runs on Ubuntu**
`README_FIRMWARE.md` provides Windows-specific instructions (Visual Studio Build Tools, Pico Windows Installer). The CI builds on Ubuntu/Linux. There are no Linux/Mac build instructions in the README. A developer on Linux following the README would find no guidance.

**VR-15 [LOW] No static analysis of firmware C code in CI**
The firmware C code (`main.c`, `usb_descriptors.c`) has no SAST step. `cppcheck`, `clang-tidy`, or even `arm-none-eabi-gcc -Wall -Wextra` warning escalation to error (`-Werror`) would catch common C errors. The current build uses whatever warning flags CMake defaults provide.

---

## CHALLENGE PHASE

Each expert challenges at least three findings.

---

**Devil's Advocate challenges OH-01 [CRITICAL — No dependency lock file]:**
"This is a research prototype, not a production service. The packaging evaluation document explicitly concluded that research velocity matters more than packaging purity. Pinning every transitive dependency would mean a package manager like `uv` or `pip-tools` must be added to the workflow, and every dependency update becomes a manual PR. For a two-person research project with no external users, unpinned dependencies with minimum version bounds is the correct trade-off. The `>=` lower bounds do protect against known regressions."

*Panel response (Lisa Park):* The Devil's Advocate is correct that full transitive pinning (pip freeze) is overkill for a research prototype. However, OH-01 is not asking for full lockfile pinning — it is asking for protection against supply chain attacks. The attack surface is real regardless of whether the project is research or production. A minimal mitigation — hashing the top-level dependencies with `pip-compile` into a `requirements.lock` file and using `pip install --require-hashes -r requirements.lock` in CI — adds almost no maintenance overhead. The finding severity should be downgraded to HIGH rather than CRITICAL for a research context, but it should not be dismissed.

*Severity adjustment: CRITICAL → HIGH for research context.*

---

**Devil's Advocate challenges VR-01 [CRITICAL — pico-sdk on master branch]:**
"The firmware build is a proof-of-concept with explicitly documented narrow scope ('transport correctness and reproducible payloads, not yet full on-device parity'). The scope decision document acknowledges the firmware is not the primary deliverable. Pinning to a specific pico-sdk release tag adds maintenance burden every time RPi Foundation releases an SDK update. For a POC that produces a demonstration UF2, using `master` is acceptable."

*Panel response (Victor Reyes):* The scope limitation is acknowledged but the reproducibility argument stands independently of scope. Even a POC should be reproducible — that is, the same source code should produce the same binary. Using `--branch master` violates this property by definition. The fix (replacing `--branch master` with `--branch 2.1.0`) is a single-line change with zero ongoing maintenance burden once done. The finding remains HIGH (adjusted from CRITICAL given the POC context).

*Severity adjustment: CRITICAL → HIGH for POC context.*

---

**Devil's Advocate challenges LP-02 [HIGH — No job dependency ordering]:**
"The argument that lint should block tests is debatable. Parallel execution means faster total wall-clock time. A developer gets test results AND lint results simultaneously, which is more useful. Forcing `test` to `needs: lint` would mean a developer waits for lint to finish (1-2 min) before tests start running. For a project with a 5-10 minute test suite, this doubles the minimum feedback time."

*Panel response (Thomas Eriksen):* The wall-clock argument is valid. However, the finding also identifies that `computational-storage-emulation` should `needs: computational-storage-fundamentals`. The lint dependency is debatable (DA makes a fair point), but the emulation job running before fundamentals are validated is a legitimate ordering concern. The finding is more nuanced than the original text implies. Recommend splitting: lint-before-test dependency is LOW, but emulation-after-fundamentals is MEDIUM.

*Partial concession: LP-02 split into lint ordering (LOW) and emulation ordering (MEDIUM).*

---

**Devil's Advocate challenges NC-01 [HIGH — 7 modules missing from py-modules]:**
"The packaging evaluation explicitly concluded that for a research prototype with no distribution intent, the `py-modules` list being slightly stale is a known and accepted trade-off. No one installs this package from PyPI. The seven missing modules are fully importable in CI and locally because the flat layout puts them on `sys.path` automatically. This is a paper finding with zero real-world impact."

*Panel response (Nina Chen):* Partially conceded. The seven missing modules are importable in development and CI because of the flat layout. The impact is only felt if `pip install chelatedai` is used (which never happens currently). However, the finding should remain HIGH because the `py-modules` list is the single source of truth for what constitutes the package, and it is demonstrably incomplete. A CI check validating list completeness would cost nothing to implement and would prevent this from silently getting worse.

*Severity maintained: HIGH — the gap will widen over time without a check.*

---

**Devil's Advocate challenges OH-02 [CRITICAL — Action SHA pinning]:**
"SHA pinning for GitHub Actions is a best practice for enterprise environments but is operationally burdensome for small research projects. SHA-pinned actions require manual updates via Dependabot or manual research to find the new SHA when a security patch is released. `actions/checkout@v4` from the GitHub-owned `actions` organization is effectively trusted. Treating it as an untrusted supply chain input is security theater for a research prototype."

*Panel response (Omar Hassan):* The enterprise vs. research distinction is acknowledged. However, the Codecov supply chain compromise (2021) used a trusted CI action to exfiltrate secrets. The risk is real. That said, the impact for this repository is low — there are no CI secrets, no deployment credentials, and no sensitive data. The pragmatic mitigation is to enable Dependabot for GitHub Actions (which auto-generates SHA-update PRs) rather than manually pinning. This addresses the spirit of the concern with minimal burden.

*Severity adjustment: CRITICAL → MEDIUM for research context with no CI secrets.*

---

**Devil's Advocate challenges TE-01 [HIGH — torch not in requirements.txt]:**
"PyTorch is intentionally excluded from `requirements.txt` because it requires a specific index URL (`--index-url https://download.pytorch.org/whl/cpu`) that cannot be expressed in a plain `requirements.txt` file without breaking other pip installs. The separate install step is the standard pattern for PyTorch in CI. The double-install concern is addressed by pip's resolver, which will recognize the already-installed version and skip re-downloading."

*Panel response (Thomas Eriksen):* The index URL constraint is a valid reason why torch is not in `requirements.txt`. The double-download concern is partially mitigated by pip's conflict resolution. However, the cache miss issue remains: the pip cache key hashes `requirements.txt`, which does not include torch. This means the 180MB torch download happens on every run regardless of whether the cache is hit. The specific solution is to add an explicit `actions/cache` step for the torch download using the PyTorch index URL as the cache key, rather than relying on `setup-python`'s built-in pip cache. Finding severity adjusted to MEDIUM.

*Severity adjustment: HIGH → MEDIUM.*

---

**Devil's Advocate challenges OH-08 [MEDIUM — nul file committed to git]:**
"The `nul` file is excluded by `.git/info/exclude` which means git knows about it and treats it as an exclude. The file is 46 bytes and has no practical impact. Removing it from git history requires a `git filter-branch` or `git filter-repo` operation that rewrites history, which is a much more disruptive change than the issue warrants."

*Panel response (Omar Hassan):* Partially conceded. The `nul` file is in `.git/info/exclude` which means it WON'T be staged or committed going forward. But `git ls-files nul` confirms it IS already tracked in the repository — it was committed at some point. `.git/info/exclude` only prevents untracked files from showing up as untracked; it does not exclude already-tracked files from commits. The file should be removed with `git rm nul` and committed. This does not require history rewriting — just a simple removal commit.

*Severity maintained: MEDIUM — easy to fix, confirmed to be tracked.*

---

**Devil's Advocate challenges VR-03 [HIGH — No HIL testing]:**
"Hardware-in-the-loop testing requires a physical RP2040 device attached to a CI runner. GitHub-hosted runners do not have USB devices. Self-hosted runners would require a dedicated hardware setup. For a research POC, the software emulation layer provides sufficient confidence. The BUILD_GUIDE.md explicitly documents that the firmware build proves transport correctness, not full parity. Requiring HIL CI for a POC is unrealistic."

*Panel response (Victor Reyes):* Fully conceded for the current scope. HIL testing is not feasible on GitHub-hosted runners. However, the finding should remain documented as a known gap — not as an immediate action item but as a gate that must be satisfied before the firmware claims are expanded beyond transport POC. The finding is reclassified as a documented gap rather than an active CI deficiency.

*Severity adjustment: HIGH → LOW (documented gap, not actionable without dedicated hardware runner).*

---

## CONVERGE — Stack-Ranked Findings

### Critical (1 finding after challenge adjustments)

| ID | Finding | Expert |
|----|---------|--------|
| ~~OH-01~~ | (Downgraded to HIGH) | — |
| ~~OH-02~~ | (Downgraded to MEDIUM) | — |
| ~~VR-01~~ | (Downgraded to HIGH) | — |

*Note: After challenge phase, no findings remain at Critical severity for this research prototype context. The panel agrees that the Critical designation requires immediate production impact. The closest candidates — unpinned dependencies and mutable SDK HEAD — are HIGH severity in a research context.*

### High Severity

| ID | Finding | Expert | Short Description |
|----|---------|--------|------------------|
| OH-01 | Supply chain: No dependency lock file | Omar Hassan | All deps unpinned; silent version drift |
| OH-03 | No explicit `permissions:` block in CI | Omar Hassan | Default write perms on all jobs |
| OH-04 | No Dependabot for dependency CVEs | Omar Hassan | No automated vulnerability PRs |
| LP-01 | torch double-install / cache miss | Lisa Park | torch in requirements.txt AND separate step |
| LP-02a | Emulation job not ordered after fundamentals | Lisa Park/TE | Missing `needs:` for emulation job |
| LP-03 | Computational storage tests run 3–5× per PR | Lisa Park | Test discovery + 2 dedicated jobs overlap |
| NC-01 | 7 production modules missing from py-modules | Nina Chen | `pip install` omits ~25% of modules |
| NC-02 | requirements.txt and pyproject.toml diverged | Nina Chen | Two sources of truth with different deps |
| VR-01 | pico-sdk cloned at mutable `master` HEAD | Victor Reyes | Non-reproducible firmware builds |
| VR-02 | Firmware artifact not tied to commit SHA | Victor Reyes | Cannot trace artifact to source |
| VR-04 | No firmware version embedded in binary | Victor Reyes | No way to identify flashed version |
| OH-05 | fusepy is unmaintained; container is privileged | Omar Hassan | FUSE + privileged = attack surface |
| OH-06 | Benchmark backup .pt files not gitignored | Omar Hassan | Large binaries could be accidentally committed |

### Medium Severity

| ID | Finding | Expert | Short Description |
|----|---------|--------|------------------|
| LP-04 | No Windows CI runner | Lisa Park | Dev platform not in CI matrix |
| LP-05 | No pico-sdk clone caching | Lisa Park | 200MB download every firmware build |
| LP-07 | No release workflow | Lisa Park | Static version 0.1.0 forever |
| LP-08 | `run_all_tests.py` bypasses structured test output | Lisa Park | No JUnit output from harness script |
| OH-02 | Action versions not SHA-pinned | Omar Hassan | Mutable semver tags |
| OH-07 | sweep_results.json not in .gitignore | Omar Hassan | Experiment output can be accidentally committed |
| OH-08 | `nul` file tracked in git | Omar Hassan | Reserved Windows device name in repo |
| OH-09 | Docker emulator runs privileged | Omar Hassan | Container escape risk |
| OH-10 | pico-sdk at mutable master | Omar Hassan | Supply chain risk |
| TE-03 | No test timeout configured | Thomas Eriksen | Runaway tests can consume 6hr CI limit |
| TE-04 | Emulation validation runs as script, not test | Thomas Eriksen | No structured test output |
| TE-05 | `fail-fast: false` wastes CI on total failures | Thomas Eriksen | All 4 versions run even on catastrophic failure |
| TE-06 | No timeout on run_all_tests.py | Thomas Eriksen | Training scripts can hang |
| TE-07 | No Python 3.9 compatibility type checking | Thomas Eriksen | `X | None` annotations may slip through |
| TE-08 | No Python 3.13 in test matrix | Thomas Eriksen | Latest Python version not covered |
| NC-03 | No `[project.urls]` metadata | Nina Chen | pip show produces no links |
| NC-04 | No trove classifiers | Nina Chen | PyPI search would not categorize project |
| NC-05 | No `[project.authors]` | Nina Chen | No maintainer contact in package metadata |
| NC-06 | No `readme` field | Nina Chen | No long description for distribution |
| NC-07 | setuptools min version too loose | Nina Chen | Build backend can change behavior silently |
| NC-08 | CI lint installs ruff separately from dev extras | Nina Chen | CI and developers use different install mechanisms |
| VR-05 | DISK_BLOCK_COUNT=200 undocumented limitation | Victor Reyes | 100KB limit not flagged in firmware code |
| VR-06 | msc_disk consumes 38% of RP2040 RAM | Victor Reyes | No headroom for future expansion |
| VR-07 | No firmware version header | Victor Reyes | No CMake version injection |
| VR-08 | pico_sdk_import.cmake is a local copy | Victor Reyes | Can drift from SDK version |
| VR-09 | No ELF constant verification in CI | Victor Reyes | Miscompilation of constants undetected |

### Low Severity

| ID | Finding | Expert | Short Description |
|----|---------|--------|------------------|
| LP-02b | Lint job not required before test job | Lisa Park | Minor ordering preference |
| LP-06 | No container isolation for Python tests | Lisa Park | Runner pre-installed packages can interfere |
| LP-09 | `ubuntu-latest` is a moving target | Lisa Park | Runner OS version can change silently |
| LP-10 | No workflow concurrency controls | Lisa Park | Redundant CI runs on rapid pushes |
| LP-11 | ruff may use stale cached version | Lisa Park | ruff 0.4 min is months behind current |
| LP-12 | No pyproject.toml validation in CI | Lisa Park | Missing modules not caught |
| LP-13 | No documentation build pipeline | Lisa Park | Broken doc links undetected |
| LP-14 | No UF2 artifact signing | Lisa Park | Firmware artifacts are unsigned |
| LP-15 | Docker emulation image not published | Lisa Park | Every user builds image locally |
| OH-11 | No SBOM generation | Omar Hassan | No software bill of materials |
| OH-12 | SECURITY.md has minimal version matrix | Omar Hassan | No CVE response SLA |
| OH-13 | No GitHub secret scanning config | Omar Hassan | No automated secret detection |
| OH-14 | `requests` should be optional-only dep | Omar Hassan | Unnecessary install for non-Ollama users |
| OH-15 | No SAST (CodeQL/Semgrep/Bandit) | Omar Hassan | No security static analysis |
| TE-09 | Emulation job installs only numpy | Thomas Eriksen | Fragile implicit dependency assumption |
| TE-10 | No apt caching in firmware build | Thomas Eriksen | Redundant ARM GCC download |
| TE-11 | No flaky test retry mechanism | Thomas Eriksen | Single failure blocks PR merge |
| TE-12 | `make -j$(nproc)` provides no benefit | Thomas Eriksen | 2-file project gains nothing from parallel make |
| TE-13 | No CI badge in README | Thomas Eriksen | Build health not visible |
| TE-14 | `os.chdir()` in run_all_tests.py is fragile | Thomas Eriksen | Changes working dir for subsequent steps |
| TE-15 | No CI check for py-modules completeness | Thomas Eriksen | Missing modules accumulate silently |
| NC-09 | No package-data configuration | Nina Chen | Non-Python assets cannot be distributed |
| NC-10 | py-modules list does not scale | Nina Chen | Manual maintenance burden grows |
| NC-11 | langdetect not in optional-dependencies | Nina Chen | No pip install path for langdetect support |
| NC-12 | No upper bound on python_requires | Nina Chen | Future Python breaking changes unchecked |
| NC-13 | httpx in dev but usage unclear | Nina Chen | Uncertain dependency inclusion |
| NC-14 | .gitignore has .coverage but no [tool.coverage] | Nina Chen | Coverage setup is inconsistent |
| NC-15 | No editable install step in any CI job | Nina Chen | py-modules completeness never tested |
| VR-03 | No HIL testing (documented gap) | Victor Reyes | Firmware logic verified by emulation only |
| VR-10 | Debug TinyUSB build never tested | Victor Reyes | Debug build may have compilation issues |
| VR-11 | No CMakePresets.json | Victor Reyes | Build config not reproducible across envs |
| VR-12 | GIT_SUBMODULES_RECURSE FALSE in cmake | Victor Reyes | TinyUSB submodule init path inconsistency |
| VR-13 | No `make clean` step | Victor Reyes | Fails on non-ephemeral runners |
| VR-14 | README_FIRMWARE.md is Windows-only | Victor Reyes | No Linux build guidance |
| VR-15 | No C static analysis in CI | Victor Reyes | Firmware code has no SAST coverage |

---

## DISSENT LOG

**Thomas Eriksen dissents on LP-03 severity (HIGH):**
"Running computational storage tests 3–5× per PR is inefficient but not a correctness issue. The redundant runs do provide more coverage across Python versions. I would rate this MEDIUM, not HIGH. The CI minutes cost is real but manageable."

*Panel notes dissent. Finding remains HIGH because the duplication is not providing additional Python version coverage for these tests (dedicated jobs run on Python 3.11 only) and the 4-version matrix runs identical tests each time.*

**Nina Chen dissents on OH-01 being downgraded from CRITICAL:**
"For any software that gets installed via pip install — even occasionally — unpinned transitive dependencies are a CRITICAL supply chain risk. The `mteb>=1.0` package alone pulls in `beir`, `datasets`, `faiss-cpu`, and other large ML packages that could be compromised. I maintain CRITICAL for any project that has a pyproject.toml and a stated intent to be installable."

*Panel notes the dissent. The project owner should decide based on distribution intent.*

**Victor Reyes dissents on VR-03 being downgraded to LOW:**
"While HIL testing is not feasible today, it should remain on the roadmap as MEDIUM — a planned gap, not an accepted gap. The phrasing 'LOW' implies it is not important, when in fact it is the most significant gap between the firmware claim and verified behavior."

*Panel accepts partial dissent: VR-03 reclassified as MEDIUM (planned gap, roadmap item) rather than LOW.*

---

## FEASIBILITY GATE

Assessment of each High finding for implementation feasibility in a research prototype context.

| ID | Finding | Effort | Impact | Feasibility Verdict |
|----|---------|--------|--------|---------------------|
| OH-01 | No dependency lock file | S (1 day) | High | FEASIBLE — add `pip-compile` output; no architecture change |
| OH-03 | No permissions block | XS (30 min) | High | FEASIBLE — add 3 lines to each workflow |
| OH-04 | No Dependabot | XS (15 min) | High | FEASIBLE — add `.github/dependabot.yml` |
| LP-01 | torch double-install/cache | S (2 hours) | Medium | FEASIBLE — add explicit `actions/cache` for torch wheel |
| LP-02a | Missing `needs:` for emulation | XS (10 min) | Medium | FEASIBLE — add `needs: computational-storage-fundamentals` |
| LP-03 | Computational storage test duplication | S (1 hour) | Medium | FEASIBLE — exclude cs tests from main discovery or mark them skip |
| NC-01 | 7 modules missing from py-modules | XS (20 min) | High | FEASIBLE — add 7 entries to pyproject.toml |
| NC-02 | requirements.txt/pyproject.toml diverged | S (1 hour) | High | FEASIBLE — align two files and add CI validation |
| VR-01 | pico-sdk on mutable master | XS (5 min) | High | FEASIBLE — replace `--branch master` with `--branch 2.1.0` |
| VR-02 | Firmware artifact not tied to SHA | XS (20 min) | Medium | FEASIBLE — include `${{ github.sha }}` in artifact name |
| VR-04 | No firmware version in binary | S (2 hours) | Medium | FEASIBLE — add `git_hash.h` via CMake configure_file |
| OH-05 | fusepy unmaintained + privileged | M (1 week) | Medium | CONDITIONAL — replace fusepy with maintained alternative; reduce container privileges |
| OH-06 | Benchmark .pt files not gitignored | XS (10 min) | High | FEASIBLE — add pattern to .gitignore |

---

## EXECUTIVE SUMMARY — Top 10 Infrastructure Findings

The ChelatedAI infrastructure is functional for a research prototype but has accumulated meaningful technical debt across dependency management, CI configuration, packaging completeness, and firmware build reproducibility. The following ten findings represent the highest combined impact-to-effort ratio:

### 1. Seven production modules missing from `py-modules` (NC-01) — HIGH
`benchmark_beir`, `cross_lingual_distillation`, `isomer_detector`, `kalman_lr_scheduler`, `language_detector`, `sedimentation_loss`, and `topology_analyzer` are absent from `pyproject.toml`'s `[tool.setuptools] py-modules`. Any `pip install chelatedai` omits ~25% of the module surface. Fix: add 7 lines to `pyproject.toml`. Effort: 20 minutes.

### 2. pico-sdk cloned at mutable `master` HEAD (VR-01 / OH-10) — HIGH
Every firmware CI build clones `--branch master` which is a moving target. Identical source code produces different binaries on different days. Fix: replace with a pinned release tag (e.g., `--branch 2.1.0`). Effort: 5 minutes.

### 3. No dependency lock file — supply chain unpinned (OH-01) — HIGH
All Python dependencies use minimum-version bounds only. Every CI run can resolve different transitive dependency versions. Fix: introduce `pip-compile` with a `requirements.lock` file. Effort: 1 day.

### 4. Computational storage tests run 3–5× per PR (LP-03) — HIGH
Main test discovery + dedicated CI jobs create redundant test execution. Fix: exclude `test_computational_storage_*.py` from the main discover job or use `--ignore` flag. Effort: 1 hour.

### 5. No explicit `permissions:` block in workflows (OH-03) — HIGH
CI jobs default to repository read/write permissions. Fix: add `permissions: {contents: read}` to each job. Effort: 30 minutes.

### 6. Benchmark `.pt` backup files not gitignored (OH-06) — HIGH
Three multi-megabyte PyTorch checkpoint files are in the working tree with no `.gitignore` coverage. Fix: add `adapter_weights.benchmark-backup-*.pt` to `.gitignore`. Effort: 10 minutes.

### 7. `requirements.txt` and `pyproject.toml` dependencies diverged (NC-02) — HIGH
`requests` and `mteb` appear in `requirements.txt` as mandatory but are optional in `pyproject.toml`. Fix: align the two files and choose one as the canonical source. Effort: 1 hour.

### 8. No Dependabot configuration (OH-04) — HIGH
No automated dependency update PRs for Python packages, GitHub Actions, or Docker images. Fix: add `.github/dependabot.yml`. Effort: 15 minutes.

### 9. torch pip caching ineffective (LP-01/TE-01) — MEDIUM
PyTorch (180MB per version) is not included in the pip cache key because it is installed outside `requirements.txt`. Fix: add explicit `actions/cache` for the PyTorch wheel directory. Effort: 2 hours.

### 10. `nul` file tracked in git (OH-08) — MEDIUM
A Windows reserved-name file is committed to the repository. On Windows systems, checkout behavior for a file named `nul` is undefined. Fix: `git rm nul && git commit -m "chore: remove accidentally committed nul device file"`. Effort: 5 minutes.

---

## INFRASTRUCTURE REMEDIATION ROADMAP

### Sprint 1 — Quick Wins (total effort: ~3 hours, zero risk)

These changes touch only configuration files and carry no risk of breaking tests or the codebase:

1. **NC-01** — Add 7 missing modules to `pyproject.toml` `py-modules` list
2. **VR-01** — Pin pico-sdk to `--branch 2.1.0` in `build_firmware.yml`
3. **OH-03** — Add `permissions: {contents: read}` to both workflows
4. **OH-06** — Add `adapter_weights.benchmark-backup-*.pt` to `.gitignore`
5. **OH-07** — Add `sweep_results.json` to `.gitignore`
6. **OH-08** — Remove `nul` file: `git rm nul`
7. **VR-02** — Include `${{ github.sha }}` in firmware artifact name
8. **LP-02a** — Add `needs: computational-storage-fundamentals` to emulation job
9. **OH-04** — Create `.github/dependabot.yml` with pip + actions + docker update intervals

### Sprint 2 — Packaging & Dependency Hygiene (total effort: ~1 day)

10. **NC-02** — Align `requirements.txt` with `pyproject.toml`; move `requests` and `mteb` to optional extras only; remove duplicates
11. **NC-01** — Add CI validation step: `python -c "import os; ..."` to verify py-modules completeness
12. **NC-03/04/05/06** — Add `[project.urls]`, classifiers, authors, readme to `pyproject.toml`
13. **LP-03** — Exclude computational storage tests from main discovery or restructure test jobs to avoid duplication
14. **OH-01** — Introduce `pip-compile` and commit `requirements.lock`; update CI to use `--require-hashes`

### Sprint 3 — CI Reliability & Efficiency (total effort: ~2 days)

15. **LP-05** — Cache pico-sdk clone with `actions/cache` keyed on SDK version tag
16. **TE-01** — Add `actions/cache` step for PyTorch wheel directory
17. **TE-03** — Add per-job timeout (`timeout-minutes: 30`) to all CI jobs
18. **TE-08** — Add Python 3.13 to test matrix (verify torch/sentence-transformers support first)
19. **LP-10** — Add `concurrency:` blocks to both workflows to cancel superseded runs
20. **LP-11** — Pin ruff to a specific version in CI lint step
21. **NC-08** — Replace `pip install ruff>=0.4` with `pip install -e ".[dev]"` in lint job

### Sprint 4 — Firmware Quality (total effort: ~3 days)

22. **VR-04** — Add git SHA injection via CMake `configure_file` into a `version.h` header
23. **VR-07** — Create `CMakePresets.json` with release and debug configurations
24. **VR-09** — Add ELF constant verification step in firmware CI: `arm-none-eabi-objdump` check
25. **VR-15** — Add `cppcheck` or `arm-none-eabi-gcc -Wall -Wextra -Werror` to firmware build
26. **VR-10** — Add parallel firmware debug build step to CI

### Sprint 5 — Security Posture (total effort: ~1 day)

27. **OH-02** — Pin GitHub Action versions to full commit SHAs (or enable Dependabot Actions updates)
28. **OH-05** — Replace `fusepy` with `libfuse` Python bindings; update Docker to drop privileged mode
29. **OH-09** — Replace `privileged: true` in docker-compose.yml with `--cap-add SYS_ADMIN --device /dev/fuse`
30. **OH-11** — Add SBOM generation step to firmware workflow (`cyclonedx-py`)
31. **OH-15** — Add `bandit` or CodeQL scan to `test.yml`

### Sprint 6 — Long Horizon (to be scheduled)

32. **LP-07** — Implement release workflow with semantic versioning
33. **VR-03** — Evaluate self-hosted runner with RP2040 attached for HIL validation
34. **LP-04** — Add Windows CI runner for Windows-specific code paths (hardware evidence path tests)
35. **NC-10** — Evaluate migration to `src/` layout if module count exceeds 40

---

## TOTAL FINDINGS SUMMARY

| Severity | Count (after challenge phase) |
|----------|-------------------------------|
| Critical | 0 (downgraded during challenge) |
| High     | 13 |
| Medium   | 25 |
| Low      | 30 |
| **Total** | **68** |

---

*Panel convened and report finalized: 2026-04-04*
*Experts: Lisa Park, Omar Hassan, Thomas Eriksen, Nina Chen, Victor Reyes, Devil's Advocate*
