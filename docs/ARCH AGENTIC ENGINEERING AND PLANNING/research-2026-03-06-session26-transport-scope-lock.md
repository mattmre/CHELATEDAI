# Research -- Session 26 Transport Scope Lock

## Scope
Decide whether the merged RP2040 path should still be described as a deterministic toy-graph proof or as a validated on-device workload.

## Findings
- Current repo docs already lean toward the narrower interpretation, but the boundary is spread across multiple files rather than stated once.
- Real hardware evidence is still outstanding.
- The full digits model remains validated only in the software mock path.
- Without a single explicit decision artifact, future sessions can easily overstate what the firmware path proves.

## Recommendation
- Accept a formal scope lock that keeps the RP2040 path in the deterministic transport-proof category for now.
- Centralize the promotion criteria so future sessions know exactly what evidence is required before broadening claims.
