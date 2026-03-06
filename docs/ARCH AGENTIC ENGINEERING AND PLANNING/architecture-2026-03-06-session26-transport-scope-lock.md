# Architecture -- Session 26 Transport Scope Lock

## Objective
Create a single authoritative description of the current computational-storage transport scope and point the product-adjacent docs at it.

## Design
1. Add a scope-decision document under `docs/`.
2. Update the POC and firmware docs to reference that decision.
3. Name explicit promotion gates so future sessions do not infer broader claims from unrelated validation.

## Acceptance Criteria
- There is one canonical document stating the current claim boundary.
- The POC README and firmware docs reference that canonical document.
- The current scope is described as a deterministic transport proof, not full on-device digits inference.
