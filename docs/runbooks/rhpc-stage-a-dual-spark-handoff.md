# RHPC Stage A dual-Spark handoff

**Protocol:** `CHELATEDAI-RHPC-STAGE-A-v1`

**Current execution state:** `WAITING_FOR_EGV_TERMINAL_SEAL`

**Claim boundary:** constructed-mechanism replication only; no learned-model,
compression-utility, MoE-routing, language-quality, novelty, or promotion claim.

## Why this waits

RHPC Stage A must not overlap the active EGV campaign or act on a merely verbal
handoff. Its official runner opens only after the prior campaign's exported
public projection verifies with a terminal public-chain seal. That closed
verification also covers the seal-bound signed restoration receipt.

The current EGV handoff still records unfinished exact-Spark acceptance,
publication, merge, and inference-service restoration. Therefore no RHPC
command in this document is currently authorized on either Spark.

## What is ready before the handoff

- The source is isolated on branch `codex/rhpc1-method-dev-20260825` in its own
  worktree.
- Stage A uses NumPy on CPU and neither imports Torch nor creates a GPU context.
- Official configurations are closed to run IDs `7` and `11`, 256 trials, and
  the frozen dimensions and gates in the protocol.
- The runner rejects a dirty source tree, absent output precondition failures,
  missing double opt-in, malformed predecessor digests, an unsealed public
  projection, or a non-official configuration.
- Publication is staged, fsynced, verified, and atomically renamed. Verification
  rejects noncanonical JSON, extra members, links, hard links, digest changes,
  contradictory status, and failure to regenerate the result exactly.

## Mechanical admission gate

The operator supplies these values from the private EGV closeout handoff; they
must not be copied into public repository files:

```text
RHPC_EGV_PROJECTION       local path to the exported public projection
RHPC_EGV_PUBLIC_KEY       local path to the evaluator public key
RHPC_EGV_PROTOCOL_DIGEST  exact EGV protocol SHA-256
RHPC_EGV_EVALUATOR_DIGEST exact evaluator SHA-256, when the campaign binds one
```

On each Spark, all of the following must be true before the first official
observation:

1. EGV work and its closeout are no longer running.
2. The operator-owned inference service has been restored and independently
   verified by the prior campaign owner.
3. The RHPC checkout is at the same reviewed immutable source SHA on both
   machines and `git status --porcelain=v1 --untracked-files=all` is empty.
4. The intended output directory does not exist.
5. The local public projection, public key, and trusted digests are available.
6. A runner dry refusal without the official environment opt-in returns code 2
   and creates no output.

The runner itself calls closed public replay with `require_terminal_seal=True`.
If the terminal seal or bound restoration evidence is absent or invalid, it
fails before constructing the experiment or creating the destination.

## Frozen 2 by 2 execution matrix

Run both IDs independently on both Sparks. Use an external per-host output root,
never the Git checkout or the inference-service directory.

| Spark role | Run 7 | Run 11 |
|---|---:|---:|
| `spark_trainer` | required | required |
| `spark_evaluator` | required | required |

The private inventory resolves those role names to SSH destinations. This
public runbook deliberately carries no hostnames, addresses, credentials, or
deployment paths.

For each matrix cell, from the clean RHPC source root:

```bash
export CHELATED_RHPC_STAGE_A=1
python run_rhpc_stage_a.py run \
  --spark-official \
  --allow-official \
  --run-id RUN_ID \
  --output ABSENT_EXTERNAL_OUTPUT_DIRECTORY \
  --predecessor-projection "$RHPC_EGV_PROJECTION" \
  --predecessor-public-key "$RHPC_EGV_PUBLIC_KEY" \
  --predecessor-protocol-digest "$RHPC_EGV_PROTOCOL_DIGEST" \
  --predecessor-evaluator-digest "$RHPC_EGV_EVALUATOR_DIGEST"
```

Omit the evaluator-digest option only when the frozen predecessor campaign did
not bind one. Do not change trial counts or thresholds; official mode ignores
the development trial option and constructs the frozen configuration.

Apply an operator-owned external timeout to each process. Do not respond to a
timeout by stopping, restarting, or reconfiguring the restored inference
service. Preserve any unpublished staging directory for diagnosis if the host
or filesystem fails during publication.

## Independent verification and comparison

Verify every cell on its originating host:

```bash
python run_rhpc_stage_a.py verify ABSOLUTE_ARTIFACT_DIRECTORY
```

Then copy only the two-file published artifacts through the approved evidence
transport and verify again on the receiving controller. For a given run ID,
`rhpc-stage-a.json` must be byte-identical across the two Sparks. Manifests may
differ in honest environment fields but must bind the same protocol SHA and
source Git SHA.

Stage A survives only when:

- all four artifacts verify exactly;
- all four report `SURVIVES_CONSTRUCTED_MECHANISM_GATES`;
- run 7 result bytes match across hosts;
- run 11 result bytes match across hosts; and
- no Spark or DeepSeek service mutation occurred.

Any failed gate, mismatch, timeout, incomplete artifact, or admission error is
retained as a negative or operational failure. It is not rerun with adjusted
thresholds under the same protocol ID.

## Interpretation and next phase

A clean 2 by 2 result establishes replication only for this deterministic
fixture. Stage B still requires a separately frozen learned small-MoE protocol,
route-disjoint data, matched parameter and compute budgets, preregistered
endpoints, and its own negative controls. Stage C remains a separate Qwen
micro-LoRA transfer test and cannot reuse an exhausted REPORT split.
