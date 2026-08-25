# R5 full-census paired semantic diagnostic

Status: diagnostic complete; no promotion or training-eligibility claim.

This report covers every exact-single-fence source-contract failure available when the primary campaign reached a frozen cutoff of 57 terminal requests. It is a matched failure census, not an iid sample and not a general model success-rate estimate.

## Frozen comparison

- Source commit: `8f1955abe8a214cc6f469547d3cc1bbddbf7d1ec`
- Model: `Qwen/Qwen3.5-2B-Base`
- Matched contexts: 387
- Condition A: deterministic strict-single-fence projection of the original response
- Condition B: one `source-only-prefill-v1` generation from the identical full evidence-bearing context
- Semantic evaluations: 774
- Infrastructure losses: 0

| Result | Condition A | Condition B |
|---|---:|---:|
| Semantic PASS | 0 | 0 |
| Wrong output | 309 | 309 |
| Runtime exception | 78 | 78 |
| Source-contract accepted | 387 | 387 |

Prefill repaired the response-format boundary for all 387 B generations, but it did not produce a semantic PASS. All matched pairs remained failures with exact diagnostic agreement: 309 wrong-output pairs and 78 runtime-exception pairs.

## Source diversity

Every A/B pair had different source bytes. Across the census there were 19 unique A source digests and 19 unique B source digests, with zero intersection and 38 in the union.

| Task family | Contexts | A unique | B unique | Intersection | Union |
|---|---:|---:|---:|---:|---:|
| Data transform | 78 | 3 | 3 | 0 | 6 |
| Dependency contract | 77 | 3 | 3 | 0 | 6 |
| Parser edge | 75 | 4 | 4 | 0 | 8 |
| Pure function | 11 | 3 | 3 | 0 | 6 |
| Resource bound | 78 | 3 | 3 | 0 | 6 |
| State transition | 68 | 3 | 3 | 0 | 6 |

The low number of unique source digests relative to 387 contexts is consistent with highly repeated outputs in this frozen run. It does not, by itself, establish the cause of the repetition.

## Coverage

- Primary arms: B 192; D 195
- Seeds: 0 had 206 contexts; 1 had 181
- Attempt bins: 1–3 had 60; 4–6 had 86; 7–9 had 109; 10–12 had 132
- Retrieval bins: 1–3 had 155; 4–7 had 188; 8–15 had 38; 16+ had 6
- Task families: data transform 78; dependency contract 77; parser edge 75; pure function 11; resource bound 78; state transition 68

## Isolation and claim boundary

Generation and evaluation used fresh, isolated state. The generator had no hidden-evaluator access. The evaluator used a fresh signer and state with a read-only copy of the hidden seed. The primary campaign ledger/cache was not mounted for writes, and its immutable artifacts remained unchanged.

These results support a narrow conclusion: response formatting was a real transport defect, but repairing it was insufficient to recover correct programs in this frozen failure census. They do not support promotion, retroactive promotion, LoRA training eligibility, production utility, or a broader model-success claim.

## Machine-readable evidence

- R5 aggregate SHA-256: `dfff9543251c58406f12e3aa49b493dd3c852823a7061ab2364f188a50027b95`
- Spark 2 CPU/Qdrant smoke SHA-256: `c208be009d8d2c8c5fa0be1dd7dd7637e4ae942b30dc27268ff977dd87d87a0b`
- Spark 2 exact-source recovery/commissioning validation SHA-256: `38a8a19dfcd3f8b16a7551a90d7d7728d8e4cc10fa6ae791549dd4c932d99b1e`
- Validation record SHA-256: `f44e2ca70ed7ce3bfe022880d1c845b5df6221f2d0222ae2932921f42d7471dc`
- Validator SHA-256: `9f7ed92cfc186f74a22fb4c33fc492614bb17d471a1b8711640dad9d747a7ddf`

The JSON artifacts use canonical bytes, exact SHA-256 pins, and recursively checked closed object schemas. The included validator runs both the targeted public-safety scan and Gitleaks; it found no raw prompts/source fields, private identifier fields, private paths/topology, credentials, or secret findings. The redacted aggregate cannot independently recompute the private row-level campaign result.
