# EGV source-only refinement canary

## Evidence basis

The first live Qwen diagnostic reached the model and completed generation, but
the response did not satisfy the closed candidate envelope. Private structural
inspection showed a wrong JSON root plus a malformed structured suffix. The
root independently failed every candidate-contract gate. It is therefore not
safe to extract the first object or discard the suffix.

## Experimental split

The next bounded canary compares two experimental response contracts on the
same frozen Arm B context and exact public source:

1. `source-only-v1`: Qwen receives the task statement and current complete
   Python file, then emits only one complete Python module.
2. `source-only-prefill-v1`: the same contract, with the first exact source line
   supplied through the official chat template's assistant-continuation mode.

The experiments use exact rendered-chat byte digests, a sealed prompt registry,
and a sealed tokenizer chat-template digest. The trusted host—not the model—sets
the declared locus and `EXECUTE_CANDIDATE` authority, conservatively binds every
presented evidence ID, and computes the mutation digest. Candidate admission
still requires the complete response to parse as Python and to define the
function named by the public locus.

The experimental evidence API retains three distinct byte sequences from the
same sealed generation call: the exact rendered prompt, the model-decoded
continuation, and the full contract response after the declared host prefill is
reassembled. Each sequence has its own digest. Validation reconstructs the
prefilled response, reparses the entire contract response, and requires the
resulting proposal to match exactly. The API is unavailable to
`closed-json-v1`; that legacy path does not claim rendered-prompt evidence.

## Boundary

These contracts are experimental and are not wired into the current production
commissioning, private-sidecar, training-freeze, or promotion protocol. The
existing `closed-json-v1` path and its durable schemas remain compatible. A
successful canary is evidence for a separately versioned integration, not an
implicit migration.

## Two-Spark execution plan

- Stop DeepSeek once inside one bounded outer campaign and restore it once at
  every terminal path.
- Run one pinned Qwen worker per Spark concurrently, one source-only variant per
  worker, with one generation each and no retry.
- Use per-node local leases and one-use markers bound to one campaign digest,
  absolute deadlines, independent watchdog cleanup, and an all-workers-absent
  barrier before restoration.
- Give every worker sealed, pairwise-disjoint read-only source/model/entry/input
  mounts plus empty output and home leaves as the only writable mounts.
- Keep prompts and raw generations private. Publish only closed structural and
  signed evidence projections.
- Transfer only a validated trusted-envelope candidate to evaluation through a
  private atomic inbox; never transfer raw output or credentials.
- Treat custody, model-load, worker, transfer, or receipt failures as a campaign
  abort. A proposal-validation failure may only allow the other already-started
  canary to finish.

## Promotion gate

Success means a complete source-only candidate passes structural and locus
validation and reaches the signed evaluator without weakening privacy,
authority, cleanup, or restoration controls. It does not by itself establish
general capability, evaluator success, or training readiness.
