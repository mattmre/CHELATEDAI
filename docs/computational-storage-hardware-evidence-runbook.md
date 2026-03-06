# Computational Storage Hardware Evidence Runbook

## Purpose
Capture an auditable report for the merged RP2040 transport path without relying on manual copy/paste.

## Preconditions
- The RP2040 firmware from `computational_storage_poc/firmware/` is flashed to a Raspberry Pi Pico.
- The device is mounted and readable from the host.
- The host can read raw sectors from the target drive or a file-backed image.

## Command

```bash
python computational_storage_poc/capture_hardware_evidence.py <drive-number-or-path> --notes "operator notes"
```

Default output directory:

```text
docs/computational-storage-hardware-evidence/
```

## What The Tool Records
- UTC capture timestamp
- target identifier and resolved device path
- exact observed JSON payload from trigger sector `100`
- exact expected deterministic payload
- field-by-field mismatches if any
- optional Windows disk metadata when a physical drive number is supplied

## Recommended Operator Flow
1. Flash the Pico with the current firmware.
2. Identify the drive number or device path.
3. Run the evidence-capture command above.
4. Review the generated Markdown report for PASS/FAIL and mismatches.
5. Commit the resulting evidence artifact in the follow-up hardware-validation PR if the result is trustworthy.

## Notes
- If no RP2040 device is connected, do not fabricate a report. Use this runbook once hardware is available.
- The current firmware scope is the deterministic transport payload, not full on-device digits-model execution.
