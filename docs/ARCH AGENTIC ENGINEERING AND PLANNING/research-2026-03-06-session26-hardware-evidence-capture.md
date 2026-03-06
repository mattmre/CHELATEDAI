# Research -- Session 26 Hardware Evidence Capture

## Scope
Prepare the real-hardware evidence item for execution without fabricating results when no RP2040 device is currently attached.

## Findings
- The active handoff requires real RP2040 evidence, but the local hardware probe found no present RP2040 / Raspberry Pi Pico / TinyUSB device tonight.
- The existing operator path is manual: flash the firmware, identify a drive, read sector `100`, then manually inspect the returned JSON.
- The transport contract is already deterministic in software through `payload_contract.py` and `usb_host_inference.py`, which makes report automation low-risk.
- The missing piece is not computation logic; it is reproducible evidence capture and storage under `docs/`.

## Constraints
- No physical evidence can be claimed without an attached device.
- The capture flow must still work for file-backed images so it can be validated in CI/tests tonight.
- The output needs to be durable and reviewable tomorrow.

## Recommendation
- Add a reusable `capture_hardware_evidence.py` tool that:
  - reads sector `100` from a physical or file-backed target,
  - compares the observed JSON payload with the deterministic expected payload,
  - records mismatches explicitly, and
  - writes both JSON and Markdown evidence artifacts under `docs/`.
- Validate the tool tonight using a file-backed image test.
- Leave the actual physical report creation for the first session with a connected RP2040 device.
