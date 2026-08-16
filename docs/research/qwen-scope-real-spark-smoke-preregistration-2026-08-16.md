# Qwen-Scope real Spark smoke preregistration

**Protocol ID:** `CHELATEDAI-QWEN-SCOPE-REAL-SPARK-SMOKE-v1`
**Status:** preregistered, not executed
**Scientific status:** `UNCONFIRMED`
**Novelty status:** `UNCONFIRMED`

## Claim boundary

This is an integration smoke, not a ChelatedAI mechanism experiment. A pass
will prove only that one pinned Qwen base model loads, the exact declared
residual layer is hooked, its raw hidden width is 2048, one official pinned
Qwen-Scope SAE checkpoint satisfies its published tensor contract, top-k
sparse features can be computed, a fixed prompt repeats within the frozen
tolerance, and durable resource-measured artifacts are emitted.

It cannot establish chelation utility, training gain, semantic improvement,
production readiness, causal interpretation, feature quality, or novelty.

## Frozen inputs and source identity

| Field | Frozen value |
|---|---|
| Base model | `Qwen/Qwen3.5-2B-Base` |
| Model revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| Official SAE | `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100` |
| SAE revision | `027267657257a8d490296286e8fab41e1c1a1a3d` |
| Checkpoint | `layer11.sae.pt` |
| Checkpoint SHA-256 | `d1828ace348b13cca9104f61fb47672e439e963d9d5fc5496f4c6b068a06499f` |
| Layer | 11, zero-based, `model.model.layers[11]` |
| Residual hidden size | 2048 |
| SAE width | 32768 |
| Top-k | 100 |
| Repeats | two identical inference passes |
| Residual tolerance | byte-identical float32 CPU capture (`SHA-256` equal) |
| Selected-feature tolerance | identical ordered IDs and maximum value delta `<=1e-4` |

The revisions were resolved from the repositories' `HEAD` refs on 2026-08-16.
The checkpoint digest is the Hugging Face `X-Linked-ETag` SHA-256 advertised
for that exact revision and file; the runner recomputes the downloaded bytes
and refuses a mismatch. The official model card describes layers 0--23,
residual-stream hooks, `d_model=2048`, `d_sae=32768`, and top-k 100.

## Frozen procedure

1. Run on a DGX Spark with CUDA. Do not run on the Windows workstation.
2. Fail before model load unless at least 12 GiB GPU memory and 12 GiB output
   filesystem space are free.
3. Require `transformers>=4.57.0`, then load tokenizer and model at the exact model revision in BF16 evaluation
   mode. Load the exact layer-11 SAE file and verify its SHA-256 before use.
4. Register a forward hook only on `model.model.layers[11]`. Run the fixed
   prompt twice under inference mode. Immediately detach each residual to
   float32 CPU memory; do not retain the prompt in the artifact.
5. Require a rank-three residual with final dimension 2048. Apply the official
   encoder equation to the final token and select exactly 100 unique indices.
6. Apply the repeatability and resource gates exactly as frozen above.
7. Require the final run directory to be absent. On any failed predicate,
   return nonzero and retain no final PASS. On success, stage the complete
   two-file result in a sibling directory, fsync it, verify canonical bytes,
   schema, safe paths, byte size, SHA-256, duplicated status and source
   identity, retain the pre-persistence and precommit timing checkpoints,
   rebuild and reverify the final manifest, check the live deadline immediately
   before promotion, then atomically rename the whole directory. The renamed
   directory remains uncommitted until parent fsync succeeds, artifact and
   manifest bindings are rehashed, a hidden receipt candidate is fully
   serialized/written/fsynced, the cooperative deadline still passes, and that
   prepared receipt appears atomically as `COMMIT.json`.

## Resource and operational gates

- Maximum process peak RSS: 32 GiB.
- Cooperative wall-time ceiling: 1,800 seconds. It includes model/SAE work,
  measurements, staged JSON writes, staged verification, and staged/parent
  fsync. The runner checks it after measurements, after the initial staged set,
  after the final manifest rebuild/reverification, and immediately before
  promotion. After promotion it prepares and fsyncs the receipt candidate and
  performs one final cooperative checkpoint before atomic receipt publication.
  The Python gate is cooperative only through that pre-receipt checkpoint; it
  does not claim the later receipt-publication instant is within 1,800 seconds.
  Only the external supervisor enforces a hard total.
- The runner records peak RSS, peak CUDA allocated/reserved memory, device
  identity, free-resource preflight values, and elapsed time.
- Real execution requires both the CLI flag and environment opt-in. The
  `--offline` option fails closed unless every pinned input is already cached.
- The artifact contains only the fixed prompt's SHA-256, not prompt text.

## Spark command

Online first run:

```bash
CHELATED_QWEN_SCOPE_REAL_SMOKE=1 timeout --signal=TERM --kill-after=30s 1900s \
  python qwen_scope_real_smoke.py \
  --allow-real-model \
  --output-dir artifacts/method-dev/qwen-scope-real-spark-smoke/run-001
```

Cache-only replay adds `--offline`. The external `timeout` is part of the
official execution procedure because the Python deadline is cooperative. It
sends `TERM` at 1,900 seconds and escalates to `KILL` 30 seconds later, so the
nominal external process bound is 1,930 seconds. The 100 seconds between the
1,800-second cooperative gate and `TERM` is supervisor enforcement lag, and
the following 30 seconds is escalation lag. Neither interval is guaranteed
persistence margin or additional experiment time.
Verify retained bytes independently with:

```bash
python qwen_scope_real_smoke.py \
  --verify-result-dir artifacts/method-dev/qwen-scope-real-spark-smoke/run-001
```

That live verifier intentionally binds `COMMIT.json` to the committed absolute
target. A byte-for-byte copy at another path must fail live verification. For
transported evidence, use the separate archive-custody mode:

```bash
python qwen_scope_real_smoke.py \
  --verify-archived-copy-dir /path/to/copied/run-001
```

Archive mode still requires exactly the canonical artifact, manifest, and
receipt with no extra members or symlinks. It validates every embedded
artifact/manifest size and digest, the lifecycle nonce, receipt semantics,
source identity, deadline predicate, and safe recorded source-target fields.
It reports the verified artifact and manifest byte sizes and SHA-256 values,
and computes and reports the canonical receipt's byte size and SHA-256.
Because the unchanged v1 schema has no external self-digest for the receipt,
that reported receipt digest is a custody value, not an independently anchored
signature. Its terminal status is `ARCHIVED_COPY_VERIFIED`, with
`current_directory_lifecycle_pass=false`; it never asserts that the copy's
current directory underwent the original commit lifecycle.

## Disposition

- **PASS:** all frozen predicates pass, the sole retained observation
  artifact's size and SHA-256 verify, the canonical manifest repeats the
  frozen status and source identity, and `COMMIT.json` binds the artifact,
  manifest, lifecycle ID, final target identity, and truthful pre-receipt
  cooperative-deadline predicate.
- **FAIL:** any predicate, resource ceiling, checksum, hook, shape, sparsity,
  repeatability, deadline, or persistence step fails.
- **INCONCLUSIVE:** infrastructure loss outside the runner (power, SSH, host
  reboot) prevents a terminal runner result. Do not convert this to PASS.
  A complete-looking hidden sibling `.stage-*` directory left by a hard stop
  is still INCONCLUSIVE; the public verifier and verifier CLI reject it. Only
  the private immediate pre-promotion byte check may inspect staging content.

The no-replace rename makes the verified two-file set visible but deliberately
does **not** commit PASS. The runner then fsyncs the parent, recomputes bindings,
records `pre_receipt_checkpoint_elapsed_seconds`, fully serializes/writes/fsyncs
a hidden receipt candidate, checks the live cooperative clock against 1,800
seconds, and atomically publishes that prepared candidate as `COMMIT.json`.
Receipt publication is the final lifecycle operation. It is intentionally not
followed by another Python deadline check, because such a check could make the
runner fail after the verifier can already observe PASS. A delayed final atomic
publication is therefore bounded only by the external supervisor, not by a
claim that receipt publication occurred before the Python ceiling. Receipt
appearance is the single terminal commit point. A stop before receipt leaves
either a hidden stage or a receipt-less final directory; both are
INCONCLUSIVE and the public verifier rejects them. A stop after receipt leaves
the complete three-file set and the public verifier can validate PASS. There
is no rollback or cleanup attempt after rename: parent-fsync, binding, deadline,
or receipt-write failure leaves the receipt absent rather than risking an
exception while a public PASS remains.

The lifecycle ID, exact byte bindings, and resolved-target hash prevent an
orphan from becoming PASS through a simple rename or receipt copied from a
different run/target. They are local integrity checks, not a signature or
cryptographic authority: a party able to edit every local byte can forge a new
self-consistent set, so independent custody is still required for adversarial
provenance claims.

No result from this smoke may promote or reject a ChelatedAI theory lane.

## Primary sources

- Official SAE model card and tensor contract:
  <https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100>
- Pinned base-model tree:
  <https://huggingface.co/Qwen/Qwen3.5-2B-Base/tree/b1485b2fa6dfa1287294f269f5fb618e03d52d7c>
- Pinned SAE tree:
  <https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100/tree/027267657257a8d490296286e8fab41e1c1a1a3d>
