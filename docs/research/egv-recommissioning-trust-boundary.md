# EGV recommissioning trust boundary

This note defines the production inputs required by
`egv.recommissioning.client` and `egv.recommissioning.protocol`. It contains no
deployment addresses, credentials, host labels, private paths, task identities,
prompts, or candidate source.

## Authority chain

Production launch authority is an Ed25519-signed
`egv-trusted-launch-receipt-v2` record. The controller key is pinned out of band.
The signed record binds the complete model-profile digest, runtime-config
digest, a fresh 256-bit service nonce, an ephemeral service public key, and the
generation-receipt public key. Both key IDs must be content-derived from their
raw Ed25519 public keys. A self-consistent record without a valid signature from
the pinned controller key is rejected.

The model service must expose one signed
`egv-recommissioning-service-observation-v1` record on the closed attestation
route used by the loopback client. The ephemeral service key signs the complete
model profile plus the launch receipt digest, runtime-config digest, and service
nonce. The model profile freezes repository identity, revision, weights,
license, engine image, precision, KV-cache type, context limit, tokenizer asset
inventory, tokenizer implementation, and chat template. Its public form is a
closed record and is privacy-scanned before projection. The client also checks
the health and single-model responses and binds their exact byte digests into
`egv-recommissioning-service-attestation-v3`.

Generation receipts use `egv-recommissioning-public-generation-receipt-v4`.
They are accepted only when their Ed25519 signature verifies under the
controller-pinned generation key and their content digest, task pseudonym,
request, service, tokenizer, budget, token accounting, and coordinate debit all
revalidate. The receipt reconstructs the complete `TokenBudget`, binds every
limit and the sequential debit count, and signs total elapsed time. The single
deadline begins before re-attestation and covers re-attestation, durable ledger
debit, the completion POST, response validation, and receipt construction.

Evaluator feedback uses
`egv-recommissioning-signed-evaluator-receipt-v1`. The receipt binds the frozen
coordinate, task package, model profile, evaluator-profile digest, attempt, prior prompt, generation
receipt, candidate, source, diagnostic, decision, and digest of the public
summary. The evaluator key ID and generation-receipt key ID are part of the
coordinate identity; supplying a different valid key and a matching signature
does not change the frozen authority. Feedback construction also revalidates
the complete controller-rooted service attestation and exact observation,
runtime, launch, service-key, controller-key, and generation-key bindings.

## Prompt and tokenizer provenance

Only a package that is an exact member of a fully validated 170-task
learnable-v2 corpus manifest may use the production adapter kind. The retained
proof binds the full canonical manifest, exact task index, task record, package,
and corpus digest, and is revalidated whenever the task is used. No process-local
underscore token or object-identity capability supplies authority. `render_prompt`
is the canonical renderer, while the production client independently rejects
the explicit test-only adapter kind.

`PinnedTokenizerRenderer.from_local_tokenizer` computes the tokenizer pin from
a recursively enumerated, sorted manifest of every regular file under the
tokenizer root. Caller-supplied names, when present, must equal that exhaustive
inventory; symlinks, non-regular entries, omitted files, and bounded-size
violations fail closed. Each entry binds its relative name, size, and SHA-256
bytes. A separate implementation digest binds the tokenizer class, concrete
`apply_chat_template` implementation, and package version. The chat-template
pin is SHA-256 over its exact UTF-8 bytes. Every
preflight invokes `apply_chat_template` for both rendered text and token IDs,
retains the exact immutable outputs privately, and revalidates the local assets
before use. The test-only renderer capability cannot enter a production
request.

## Public projection

Public generation records use a content-derived `task_<sha256>` reference, not
the raw task identity. Candidate source, prompts, raw responses, evaluator
details, and feedback text remain private. Public feedback contains the summary
digest only. Public projectors reject credential-shaped text, private-key
markers, private path forms, addresses, endpoints, email-shaped identifiers,
and numbered topology labels before returning a record. Opaque cryptographic
digests and signatures are validated structurally and are not interpreted as
free text.

The real learnable-v2 validator is an exact-tree integration dependency. Tests
import it directly and build a genuine model-visible package; module injection
or a permissive stand-in is not an accepted test path.

## Runtime integration boundary

This slice establishes local contract readiness only. It does not provide the
controller-side production launch-receipt issuer, a model-service implementation
of the signed attestation route, deployment wiring, or evidence of a live model
call. Those are explicit runtime integration requirements and must be proven on
the exact integrated tree before any recommissioning or research-success claim.
