# Architecture -- Session 26 Emulator CI

## Objective
Add a visible CI gate for emulator-path correctness without making GitHub Actions depend on FUSE or privileged Docker.

## Design
1. Extract the reusable emulation logic into `virtual_controller.py`.
2. Keep `fuse_block_emulator.py` as a thin adapter around that core.
3. Add `validate_emulation_path.py` to:
   - materialize a file-backed flash image,
   - read the trigger sector through `usb_host_inference.py`, and
   - assert parity with `compute_payload_result()`.
4. Add a dedicated workflow job that installs only `numpy` and runs the emulation tests and validation script.

## Acceptance Criteria
- The emulator path is covered by a dedicated CI job.
- The FUSE emulator remains usable for local/manual demos.
- The CI path does not require privileged Docker or `/dev/fuse`.

## Non-Goals
- Running the full Docker/FUSE stack in hosted CI
- Claiming physical-hardware validation
- Expanding the firmware feature scope
