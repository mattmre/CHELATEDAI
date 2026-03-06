# Research -- Session 26 Emulator CI

## Scope
Decide whether the repo should add a dedicated emulator-path CI check beyond the existing firmware build and Python validation.

## Findings
- The current repo already validates:
  - the deterministic payload contract (`test_computational_storage_payload.py`),
  - the software-only block-graph fundamentals (`test_computational_storage_poc.py`), and
  - RP2040 firmware compilation (`build_firmware.yml`).
- The existing Docker/FUSE emulator is useful for manual demonstration but is a poor default GitHub Actions target because it depends on privileged `/dev/fuse`.
- The emulator’s real value is the interception semantics, not the FUSE dependency itself.
- Those semantics can be extracted into a pure-Python controller and tested directly.

## Recommendation
- Add a separate CI gate that validates the emulation path without FUSE:
  - extract a dependency-light virtual controller,
  - validate that it materializes the same sector `100` payload,
  - validate host-reader interoperability against that materialized image, and
  - keep the Docker/FUSE path available for manual demos only.

## Why This Is Worthwhile
- It turns “should we add emulator CI?” from a docs question into a concrete, cheap coverage gain.
- It keeps the transport contract aligned across payload builder, host reader, and emulator semantics.
- It avoids flaky privileged-container behavior in hosted CI.
