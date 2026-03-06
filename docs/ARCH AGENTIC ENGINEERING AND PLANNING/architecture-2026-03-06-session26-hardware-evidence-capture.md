# Architecture -- Session 26 Hardware Evidence Capture

## Objective
Convert the hardware-evidence task from an informal manual procedure into a deterministic capture workflow that can be executed immediately once hardware is available.

## Design
1. Reuse `usb_host_inference.py` for the raw sector read.
2. Add a small automation layer that:
   - resolves target identity,
   - captures the observed payload,
   - compares it to `compute_payload_result()`,
   - records optional device metadata, and
   - writes machine-readable plus human-readable reports.
3. Validate the automation against a temporary file image populated at trigger sector `100`.

## Deliverables
- `computational_storage_poc/capture_hardware_evidence.py`
- `test_computational_storage_hardware_evidence.py`
- Research and architecture artifacts for auditability

## Non-Goals
- Fabricating a real-hardware result when no RP2040 is attached
- Expanding firmware scope beyond the current deterministic transport payload
- Deleting or altering backup refs/artifacts

## Acceptance Criteria
- A single command can capture an evidence report from a drive number, device path, or file-backed image.
- The report clearly states PASS/FAIL against the deterministic payload contract.
- Tests validate successful capture and mismatch reporting without physical hardware.
