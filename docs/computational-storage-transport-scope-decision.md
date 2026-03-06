# Computational Storage Transport Scope Decision

- Status: Accepted
- Date: 2026-03-06
- Owner: Session 26 implementation follow-up

## Current Claim Boundary

The merged RP2040 transport path is currently a **deterministic transport proof**, not a validated on-device digits-model workload.

What is in scope today:
- deterministic JSON payload generation from sector `100`
- parity between the payload contract, host reader, emulator semantics, and firmware transport surface
- software-only validation of the real digits model in the computational-storage mock path
- RP2040 firmware build correctness

What is explicitly out of scope today:
- claiming the full trained digits model already fits and executes on-device on the RP2040
- claiming real hardware latency or throughput beyond software and theoretical validation
- presenting the firmware payload as equivalent to the software digits-model evaluation

## Promotion Gates

The transport path should remain scope-locked to the deterministic toy-graph proof until all of the following are true:

1. A real RP2040 hardware evidence report is captured and committed.
2. The dedicated emulator-path CI check is green on the current branch.
3. The RP2040 firmware build is green on the current branch.
4. A concrete on-device workload is defined, including model shape, resource assumptions, and acceptance criteria.
5. A workload-specific validation artifact is added that proves the promoted claim, rather than inferring it from the existing toy payload.

## Decision

Until those gates are satisfied, contributors should describe the current RP2040 path as:

> A deterministic transport/control-plane proof that validates trigger-sector interception and payload consistency.

They should not describe it as full on-device digits inference.
