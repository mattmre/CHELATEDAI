"""Safe Slice 2 command-line paths for evidence storage and replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from .canonical import canonical_json, digest_bytes, digest_for
from .campaign.commissioning import prepare_commissioning
from .campaign.errors import CampaignError
from .campaign.runner import freeze_commissioning_dataset, run_commissioning
from .evaluation.artifacts import freeze_evaluation
from .evaluation.dataset import EvaluationCorpus
from .evaluation.smoke import run_evaluation_smoke
from .errors import EGVError, PhaseUnavailable
from .ledger import EvidenceLedger
from .projection import InMemoryProjection
from .public import (
    PublicCryptographicVerifier,
    PublicEventChain,
    PublicProjection,
    build_public_restore_receipt,
    load_public_projection,
    public_candidate_digest,
    public_dependency_set_digest,
)
from .pilot import run_two_process_smoke
from .receipts import ReceiptSigner
from .experiment import (
    CoordinateOperationStore,
    FrozenHeldoutProtocol,
    HeldoutJournal,
    HeldoutProtocolError,
    HeldoutVerifierServiceManifest,
    build_private_campaign_record,
    build_trainer_evidence_package,
    ordered_public_heldout_task_records,
    run_heldout_verifier_once,
    run_pending_coordinates,
)
from .training import (
    freeze_external_development_service,
    receive_external_adapter,
    run_external_evaluator_once,
    run_production_training,
    run_training_smoke,
)
from .variation import (
    MODEL_REVISION,
    PinnedModelLoader,
    VARIATION_PROTOCOL_DIGEST,
    build_remote_evaluator_service_manifest,
    run_remote_evaluator_once,
    run_variation_smoke,
)
from .variation.generator import model_generation_profile_digest
from .variation.remote import REMOTE_VARIATION_REQUEST_LIMIT


SLICE2_PHASES = {
    "preflight",
    "capture-services",
    "stop-services",
    "stage",
    "freeze",
    "generate-trajectories",
    "evaluate",
    "redact-and-package",
    "restore-services",
}

HELDOUT_CLI_REQUEST_LIMIT = 1_048_576
HELDOUT_JSON_LIMIT = 4_194_304


def _fail_closed_integration(*_args: Any, **_kwargs: Any) -> Any:
    raise PhaseUnavailable("the selected held-out integration is intentionally unavailable")


# Public CLI names are closed. Deployment-specific integrations must be added
# to reviewed source; a user-controlled module path is never imported.
HELDOUT_OBSERVATION_VERIFIERS = {"fail-closed-v1": _fail_closed_integration}
HELDOUT_RECONCILIATION_PROBES = {"fail-closed-v1": _fail_closed_integration}
HELDOUT_COORDINATE_RUNNERS = {"fail-closed-v1": _fail_closed_integration}
HELDOUT_RESULT_VERIFIERS = {"fail-closed-v1": _fail_closed_integration}
HELDOUT_RECONCILERS = {"fail-closed-v1": _fail_closed_integration}


def _sha(label: str) -> str:
    return digest_for(label)


def run_core_smoke() -> Dict[str, Any]:
    """Run a deterministic, dependency-light production-path evidence smoke."""

    with tempfile.TemporaryDirectory(prefix="egv-smoke-") as directory:
        root = Path(directory)
        ledger_path = root / "ledger.sqlite"
        blob_root = root / "private"
        ledger = EvidenceLedger(
            ledger_path,
            blob_root=blob_root,
            clock=lambda: "2026-08-21T00:00:00Z",
        )
        try:
            campaign_id = "campaign-smoke"
            run_id = "run-smoke"
            task_id = "task-smoke"
            candidate_id = "candidate-smoke"
            protocol_digest = _sha("egv-smoke-protocol-v1")
            evaluator_digest = _sha("egv-smoke-evaluator-v1")
            policy_digest = _sha("egv-smoke-policy-v1")
            artifact_digest = _sha("egv-smoke-candidate-artifact")
            ledger.create_campaign(
                campaign_id,
                protocol_hash=protocol_digest,
                source_commit="commit-smoke",
                model_revision="model-smoke",
                data_manifest_hash=_sha("egv-smoke-data"),
                evaluator_hash=evaluator_digest,
                policy_hash=policy_digest,
                seed_set=[7],
                created_at="2026-08-21T00:00:00Z",
            )
            ledger.create_run(
                run_id,
                campaign_id=campaign_id,
                arm="D",
                task_id=task_id,
                seed=7,
                parent_checkpoint=None,
                start_state="READY",
                host_role="spark_trainer",
                software_manifest_hash=_sha("egv-smoke-software"),
                created_at="2026-08-21T00:00:00Z",
            )
            ledger.append_candidate(
                candidate_id,
                campaign_id=campaign_id,
                run_id=run_id,
                task_id=task_id,
                parent_candidate_id=None,
                mutation_family="PURE_FUNCTION",
                patch_hash=_sha("egv-smoke-patch"),
                requested_authority="EXECUTE_CANDIDATE",
                prompt_hash=_sha("egv-smoke-prompt"),
                model_hash=_sha("egv-smoke-model"),
                adapter_hash=_sha("egv-smoke-adapter"),
            )
            # A fixed test-only seed makes the smoke artifact and its public
            # signatures reproducible. It is never written to a bundle or used
            # as a campaign credential.
            signer = ReceiptSigner(b"\x01" * 32)
            private_common = {
                "campaign_id": campaign_id,
                "run_id": run_id,
                "task_id": task_id,
                "candidate_id": candidate_id,
                "candidate_artifact_digest": artifact_digest,
                "protocol_digest": protocol_digest,
                "policy_digest": policy_digest,
                "evaluator_digest": evaluator_digest,
            }
            authority_receipt = signer.sign_receipt(
                {
                    **private_common,
                    "receipt_type": "AUTHORITY",
                    "request_id": "request-authority-smoke",
                    "decision": "ALLOW",
                },
                sequence=1,
                idempotency_key="idem-authority-smoke",
            )
            verdict_receipt = signer.sign_receipt(
                {
                    **private_common,
                    "receipt_type": "VERDICT",
                    "request_id": "request-verdict-smoke",
                    "decision": "PASS",
                    "diagnostic_enum": "PASS",
                    "resource_bucket": "UNDER_25",
                    "exit_status_class": "SUCCESS",
                    "output_digest": _sha("egv-smoke-output"),
                },
                sequence=2,
                previous_receipt_hash=digest_for(authority_receipt),
                idempotency_key="idem-verdict-smoke",
            )
            effect_receipt = signer.sign_receipt(
                {
                    **private_common,
                    "receipt_type": "EFFECT",
                    "request_id": "request-effect-smoke",
                    "decision": "ALLOW",
                    "normalized_action_hash": _sha("egv-smoke-action"),
                    "sandbox_id": "sandbox-smoke",
                    "started_at": "2026-08-21T00:00:00Z",
                    "finished_at": "2026-08-21T00:00:00Z",
                    "exit_status_class": "SUCCESS",
                    "output_digest": _sha("egv-smoke-effect-output"),
                    "environment_diff_digest": _sha("egv-smoke-environment-diff"),
                },
                sequence=3,
                previous_receipt_hash=digest_for(verdict_receipt),
                idempotency_key="idem-effect-smoke",
            )
            ledger.ingest_receipt(authority_receipt, signer.public_key)
            ledger.ingest_receipt(verdict_receipt, signer.public_key)
            ledger.ingest_receipt(effect_receipt, signer.public_key)
            ledger.append_verdict(
                "verdict-smoke",
                candidate_id=candidate_id,
                correctness=True,
                performance={"score": 1},
                hidden_test_set_hash=_sha("egv-smoke-hidden-tests"),
                evaluator_revision="evaluator-smoke-v1",
                receipt_id=verdict_receipt["receipt_id"],
                signed_receipt_hash=digest_for(verdict_receipt),
            )
            ledger.append_effect_receipt(
                "request-effect-smoke",
                candidate_id=candidate_id,
                identity="evaluator-smoke",
                normalized_action_hash=_sha("egv-smoke-action"),
                decision="ALLOW",
                policy_hash=policy_digest,
                sandbox_id="sandbox-smoke",
                started_at="2026-08-21T00:00:00Z",
                finished_at="2026-08-21T00:00:00Z",
                exit_status_class="SUCCESS",
                output_hash=_sha("egv-smoke-effect-output"),
                environment_diff_hash=_sha("egv-smoke-environment-diff"),
                signature=effect_receipt["signature"],
                receipt_id=effect_receipt["receipt_id"],
            )
            private_integrity = ledger.verify_integrity()
            if ledger.candidate_disposition(candidate_id) != "PROMOTED":
                raise EGVError("core smoke candidate did not promote")

            public_projection = PublicProjection()
            public_projection.set_public_key(signer.public_key_pem)
            public_candidate = {
                "campaign_id": campaign_id,
                "run_id": run_id,
                "task_id": task_id,
                "arm": "D",
                "attempt_index": 1,
                "candidate_id": candidate_id,
                "parent_candidate_id": None,
                "candidate_artifact_digest": artifact_digest,
                "model_digest": _sha("egv-smoke-model"),
                "adapter_digest": None,
                "prompt_template_digests": [_sha("egv-smoke-template")],
                "mutation_family": "PURE_FUNCTION",
                "normalized_public_locus": "module:function",
                "requested_authority": "EXECUTE_CANDIDATE",
                "declared_public_evidence_ids": [],
                "public_dependency_ids": [],
            }
            public_projection.add_candidate(public_candidate)
            candidate_digest = public_candidate_digest(public_candidate)
            dependency_digest = public_dependency_set_digest([])
            public_authority = signer.sign_public_receipt(
                {
                    "receipt_type": "AUTHORITY",
                    "campaign_id": campaign_id,
                    "run_id": run_id,
                    "task_id": task_id,
                    "request_id": "request-authority-smoke",
                    "candidate_id": candidate_id,
                    "candidate_artifact_digest": artifact_digest,
                    "protocol_digest": protocol_digest,
                    "policy_digest": policy_digest,
                    "evaluator_digest": evaluator_digest,
                    "public_candidate_record_digest": candidate_digest,
                    "public_dependency_set_digest": dependency_digest,
                    "decision": "ALLOW",
                },
                public_sequence=1,
            )
            public_verdict = signer.sign_public_receipt(
                {
                    "receipt_type": "VERDICT",
                    "campaign_id": campaign_id,
                    "run_id": run_id,
                    "task_id": task_id,
                    "request_id": "request-verdict-smoke",
                    "candidate_id": candidate_id,
                    "candidate_artifact_digest": artifact_digest,
                    "protocol_digest": protocol_digest,
                    "policy_digest": policy_digest,
                    "evaluator_digest": evaluator_digest,
                    "public_candidate_record_digest": candidate_digest,
                    "public_dependency_set_digest": dependency_digest,
                    "decision": "PASS",
                    "diagnostic_enum": "PASS",
                    "resource_bucket": "UNDER_25",
                    "exit_status_class": "SUCCESS",
                    "output_digest": _sha("egv-smoke-output"),
                },
                public_sequence=2,
                previous_public_receipt_digest=digest_for(public_authority),
            )
            public_effect = signer.sign_public_receipt(
                {
                    "receipt_type": "EFFECT",
                    "campaign_id": campaign_id,
                    "run_id": run_id,
                    "task_id": task_id,
                    "request_id": "request-effect-smoke",
                    "candidate_id": candidate_id,
                    "candidate_artifact_digest": artifact_digest,
                    "protocol_digest": protocol_digest,
                    "policy_digest": policy_digest,
                    "evaluator_digest": evaluator_digest,
                    "public_candidate_record_digest": candidate_digest,
                    "public_dependency_set_digest": dependency_digest,
                    "decision": "ALLOW",
                    "output_digest": _sha("egv-smoke-effect-output"),
                    "environment_diff_digest": _sha("egv-smoke-environment-diff"),
                    "exit_status_class": "SUCCESS",
                },
                public_sequence=3,
                previous_public_receipt_digest=digest_for(public_verdict),
            )
            public_projection.add_receipt(public_authority)
            public_projection.add_receipt(public_verdict)
            public_projection.add_receipt(public_effect)
            restore = build_public_restore_receipt(
                {
                    "campaign_id": campaign_id,
                    "logical_service_set_id": "svc-set-smoke",
                    "logical_service_ids": ["svc-001"],
                    "private_inventory_digest": _sha("egv-smoke-private-inventory"),
                    "service_definition_set_digest": _sha("egv-smoke-service-definitions"),
                    "model_set_digest": _sha("egv-smoke-model-set"),
                    "configuration_set_digest": _sha("egv-smoke-config-set"),
                    "executable_or_image_set_digest": _sha("egv-smoke-executable-set"),
                    "expected_service_count": 1,
                    "restored_service_count": 1,
                    "health_check_count": 1,
                    "health_pass_count": 1,
                    "all_health_checks_passed": True,
                    "smoke_input_digest": _sha("egv-smoke-service-input"),
                    "smoke_output_digest": _sha("egv-smoke-service-output"),
                    "smoke_matches_baseline": True,
                    "restoration_outcome": "RESTORED",
                },
                signer,
            )
            public_projection.set_restore_receipt(restore)
            chain = PublicEventChain(signer, campaign_id=campaign_id, protocol_digest=protocol_digest, evaluator_digest=evaluator_digest)
            chain.append_recorded_disposition(
                run_id=run_id,
                task_id=task_id,
                candidate_id=candidate_id,
                disposition="PROMOTED",
                candidate_digest=candidate_digest,
                dependency_digest=dependency_digest,
            )
            public_projection.add_event(chain.events[-1])
            seal = chain.seal(
                run_id=run_id,
                task_id=task_id,
                receipt_head_digest=digest_for(public_effect),
                candidate_records=[public_candidate],
                dependency_records=[],
                restore_receipt=restore,
            )
            public_projection.add_event(seal)
            report = PublicCryptographicVerifier(
                signer.public_key,
                protocol_digest=protocol_digest,
                evaluator_digest=evaluator_digest,
            ).verify(
                candidates=public_projection.candidates.values(),
                dependencies=public_projection.dependencies.values(),
                receipts=public_projection.receipts.values(),
                events=public_projection.events,
                restore_receipt=restore,
            )
            projection = InMemoryProjection(collection_name="egv-smoke")
            projection_manifest = projection.rebuild(
                ledger,
                lambda payload: [float(len(canonical_json(payload))), 1.0],
                embedding_model_revision="embedding-smoke-v1",
                campaign_id=campaign_id,
            )
            export_path = root / "ledger.jsonl"
            exported = ledger.export_jsonl(export_path)
            replay_path = root / "replayed.sqlite"
            replayed = EvidenceLedger.replay_jsonl(exported, replay_path, blob_root=root / "replayed-private")
            try:
                replay_integrity = replayed.verify_integrity()
            finally:
                replayed.close()
            return {
                "smoke": "PASS",
                "runtime_tier": "floor",
                "fixture_kind": "synthetic-deterministic-evidence-core",
                "qdrant_exercised": False,
                "campaign_path_exercised": False,
                "tier_limitations": [
                    "memory projection only; installed-Qdrant rebuild is a separate regression path",
                    "synthetic campaign fixture; no real campaign or service path",
                ],
                "ledger": {**private_integrity, "replay": replay_integrity},
                "projection": projection_manifest,
                "public_replay": report.to_dict(),
                "artifact_paths": [str(export_path.name)],
            }
        finally:
            ledger.close()


def _json_output(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    elif isinstance(value, Mapping):
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(value)


def _commissioning_output_targets(*outputs: Path) -> tuple[Path, ...]:
    """Reject overlapping frozen destinations before publishing any file."""

    targets = tuple(Path(output) for output in outputs)
    if len(targets) < 2:
        raise CampaignError("commissioning requires distinct frozen output paths")
    resolved = tuple(path.resolve(strict=False) for path in targets)
    normalized = tuple(os.path.normcase(str(path)) for path in resolved)
    if len(set(normalized)) != len(normalized) or any(path.is_symlink() for path in targets):
        raise CampaignError("commissioning outputs must be distinct non-aliased paths")
    existing = [path for path in targets if path.exists()]
    for index, left in enumerate(existing):
        if any(os.path.samefile(left, right) for right in existing[index + 1 :]):
            raise CampaignError("commissioning outputs must not be hard-linked")
    if any(path.exists() for path in targets):
        raise CampaignError("commissioning output already exists; refusing to overwrite frozen inputs")
    return targets


def _commissioning_generation_profile(model_root: Path, model_digest: str) -> str:
    root = Path(model_root)
    loader = PinnedModelLoader(root)
    manifest, _files = loader.verify_manifest()
    if manifest.digest() != model_digest:
        raise CampaignError("commissioning model root differs from the declared model digest")
    try:
        from transformers import AutoTokenizer

        with loader._offline_environment():
            tokenizer = AutoTokenizer.from_pretrained(
                str(root),
                revision=MODEL_REVISION,
                local_files_only=True,
                trust_remote_code=False,
            )
    except Exception as exc:
        raise CampaignError("commissioning tokenizer cannot be loaded from the pinned offline model") from exc
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise CampaignError("commissioning tokenizer lacks pinned chat-template bytes")
    return model_generation_profile_digest(
        "source-only-v1",
        model_manifest_digest=model_digest,
        chat_template_digest=digest_bytes(chat_template.encode("utf-8")),
        max_new_tokens=512,
    )


def _atomic_publish_new_json(path: Path, value: Mapping[str, Any]) -> tuple[int, int]:
    """Publish canonical JSON atomically without replacing an existing destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=".commissioning-output-", dir=str(path.parent))
    temporary = Path(name)
    created_identity: Optional[tuple[int, int]] = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            created_identity = _file_identity(os.fstat(handle.fileno()), "new frozen output")
        os.link(str(temporary), str(path))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    if created_identity is None:
        raise HeldoutProtocolError("new frozen output identity is unavailable")
    return created_identity


def _file_identity(file_stat: os.stat_result, label: str) -> tuple[int, int]:
    """Return a stable portable file identity or fail closed."""

    device = getattr(file_stat, "st_dev", None)
    inode = getattr(file_stat, "st_ino", None)
    if type(device) is not int or type(inode) is not int or inode == 0:
        raise HeldoutProtocolError("{} has no stable file identity".format(label))
    return device, inode


def _reject_reparse_ancestors(path: Path, label: str) -> None:
    """Reject symlink or reparse-point ancestors without resolving through them."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    absolute = Path(os.path.abspath(str(path)))
    ancestors = []
    cursor = absolute.parent
    while cursor != cursor.parent:
        ancestors.append(cursor)
        cursor = cursor.parent
    ancestors.append(cursor)
    for ancestor in reversed(ancestors):
        try:
            ancestor_stat = ancestor.lstat()
        except OSError as exc:
            raise HeldoutProtocolError("{} ancestor is unavailable".format(label)) from exc
        if ancestor.is_symlink() or bool(
            getattr(ancestor_stat, "st_file_attributes", 0) & reparse_flag
        ):
            raise HeldoutProtocolError("{} must not traverse a symlink or reparse ancestor".format(label))


def _read_regular_file(
    path: Path,
    label: str,
    *,
    limit: int,
    exact_size: Optional[int] = None,
    private: bool = False,
) -> tuple[bytes, tuple[int, int]]:
    """Read one bounded single-link regular file and return its exact identity."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    target = Path(path)
    _reject_reparse_ancestors(target, label)
    try:
        before = target.lstat()
    except OSError as exc:
        raise HeldoutProtocolError("{} is unavailable".format(label)) from exc
    if (
        target.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or bool(getattr(before, "st_file_attributes", 0) & reparse_flag)
        or (getattr(before, "st_nlink", 1) != 1)
    ):
        raise HeldoutProtocolError("{} must be a regular non-symlink file".format(label))
    if before.st_size > limit or (exact_size is not None and before.st_size != exact_size):
        raise HeldoutProtocolError("{} has an invalid bounded size".format(label))
    if private and os.name == "posix" and before.st_mode & 0o077:
        raise HeldoutProtocolError("{} permissions must deny group and other access".format(label))
    before_identity = _file_identity(before, label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(target), flags)
        try:
            opened = os.fstat(descriptor)
            opened_identity = _file_identity(opened, label)
            if (
                opened_identity != before_identity
                or not stat.S_ISREG(opened.st_mode)
                or bool(getattr(opened, "st_file_attributes", 0) & reparse_flag)
                or getattr(opened, "st_nlink", 1) != 1
                or opened.st_size > limit
                or (exact_size is not None and opened.st_size != exact_size)
                or (private and os.name == "posix" and opened.st_mode & 0o077)
            ):
                raise HeldoutProtocolError("{} changed during admission".format(label))
            chunks = []
            remaining = limit + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > limit:
                raise HeldoutProtocolError("{} exceeds the bounded size".format(label))
        finally:
            os.close(descriptor)
    except HeldoutProtocolError:
        raise
    except OSError as exc:
        raise HeldoutProtocolError("{} cannot be read safely".format(label)) from exc
    if exact_size is not None and len(raw) != exact_size:
        raise HeldoutProtocolError("{} has an invalid bounded size".format(label))
    return raw, opened_identity


def _read_regular_bytes(
    path: Path,
    label: str,
    *,
    limit: int,
    exact_size: Optional[int] = None,
    private: bool = False,
) -> bytes:
    raw, _identity = _read_regular_file(
        path, label, limit=limit, exact_size=exact_size, private=private
    )
    return raw


def _load_canonical_json(path: Path, label: str, *, limit: int = HELDOUT_JSON_LIMIT) -> Dict[str, Any]:
    """Load one bounded regular canonical JSON object without aliases."""

    raw = _read_regular_bytes(path, label, limit=limit)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise HeldoutProtocolError("{} is not readable JSON".format(label)) from exc
    if not isinstance(value, dict) or raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise HeldoutProtocolError("{} must be canonical JSON plus one newline".format(label))
    return value


def _load_heldout_protocol(path: Path) -> FrozenHeldoutProtocol:
    """Rebuild a frozen protocol and require byte-level semantic identity."""

    value = _load_canonical_json(path, "held-out protocol")
    required = {
        "schema_version", "campaign_id", "bindings", "evaluator_public_key_hex",
        "evaluator_key_id", "schedule_seed", "bootstrap_seed", "bootstrap_replicates",
        "bootstrap_method", "confidence_level", "heldout_task_ids", "seeds",
        "heldout_task_records", "heldout_task_records_digest",
        "shock_task_ids", "max_attempts", "max_post_shock_attempts", "coordinates",
        "protocol_digest",
    }
    if set(value) != required:
        raise HeldoutProtocolError("held-out protocol is not a closed object")
    protocol = FrozenHeldoutProtocol.build(
        campaign_id=value["campaign_id"],
        bindings=value["bindings"],
        evaluator_public_key=bytes.fromhex(value["evaluator_public_key_hex"]),
        schedule_seed=value["schedule_seed"],
        bootstrap_seed=value["bootstrap_seed"],
        heldout_task_records=value["heldout_task_records"],
        heldout_task_ids=value["heldout_task_ids"],
        seeds=value["seeds"],
        shock_task_ids=value["shock_task_ids"],
        bootstrap_replicates=value["bootstrap_replicates"],
    )
    if protocol.to_private_dict() != value:
        raise HeldoutProtocolError("held-out protocol differs from its deterministic reconstruction")
    return protocol


def _approved_integration(registry: Mapping[str, Any], name: str, label: str) -> Any:
    """Resolve only a source-reviewed integration name from a closed registry."""

    value = registry.get(name)
    if value is None or not callable(value):
        raise HeldoutProtocolError("{} is not an approved integration".format(label))
    return value


def _move_noreplace(source: Path, destination: Path) -> bool:
    """Move one exact filesystem object without replacing a newer destination."""

    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move_file.restype = ctypes.c_int
        if move_file(str(source), str(destination), 0):
            return True
        error = ctypes.get_last_error()
        if error in {80, 183}:  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
            return False
        raise OSError(error, "no-replace rollback restore failed", str(destination))

    if sys.platform.startswith("linux"):
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if result == 0:
                return True
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                return False
            if error != errno.ENOSYS:
                raise OSError(error, "no-replace rollback restore failed", str(destination))

    try:
        os.link(str(source), str(destination), follow_symlinks=False)
        return True
    except FileExistsError:
        return False


def _publish_json_transaction(outputs: Sequence[tuple[Path, Mapping[str, Any]]]) -> None:
    """Publish a frozen set or recoverably quarantine members on failure.

    Rollback never unlinks by pathname.  A pathname can be replaced after an
    identity check, so destructive cleanup cannot safely prove that it still
    names the inode created by this invocation.  Instead each visible output is
    atomically moved into a private, uniquely named quarantine directory.  If
    the moved inode is not ours, a no-replace hard link restores that exact
    inode at the public pathname while the quarantine copy remains recoverable.
    """

    targets = _commissioning_output_targets(*(path for path, _value in outputs))
    encoded = [(canonical_json(value) + "\n").encode("utf-8") for _path, value in outputs]
    published: list[tuple[Path, bytes, tuple[int, int]]] = []
    try:
        for target, (_declared_path, value), expected in zip(targets, outputs, encoded):
            created_identity = _atomic_publish_new_json(target, value)
            published.append((target, expected, created_identity))
            actual, identity = _read_regular_file(
                target, "new frozen output", limit=len(expected)
            )
            if identity != created_identity or actual != expected:
                raise HeldoutProtocolError("new frozen output differs after publication")
    except Exception:
        for target, _expected, created_identity in reversed(published):
            try:
                try:
                    visible_identity = _file_identity(target.lstat(), "rollback output")
                except OSError:
                    continue
                if visible_identity != created_identity:
                    # Another process owns the current pathname.  Never move,
                    # replace, or unlink it as part of our rollback.
                    continue
                quarantine_root = Path(
                    tempfile.mkdtemp(prefix=".egv-rollback-quarantine-", dir=str(target.parent))
                )
                quarantined = quarantine_root / target.name
                os.replace(str(target), str(quarantined))
                moved_identity = _file_identity(quarantined.lstat(), "quarantined frozen output")
                if moved_identity != created_identity:
                    try:
                        _move_noreplace(quarantined, target)
                    except (NotImplementedError, OSError, TypeError):
                        pass
            except (OSError, HeldoutProtocolError):
                pass
        raise


def _read_ed25519_key(path: Path, label: str, *, private: bool) -> bytes:
    key = _read_regular_bytes(path, label, limit=32, exact_size=32, private=private)
    if len(key) != 32:
        raise HeldoutProtocolError("{} must be an exact raw Ed25519 key".format(label))
    return key


def _read_bounded_stdin_json(label: str) -> Dict[str, Any]:
    stream = getattr(sys.stdin, "buffer", sys.stdin)
    raw = stream.read(HELDOUT_CLI_REQUEST_LIMIT + 1)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if len(raw) > HELDOUT_CLI_REQUEST_LIMIT:
        raise HeldoutProtocolError("{} exceeds the bounded input limit".format(label))
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise HeldoutProtocolError("{} is not valid JSON".format(label)) from exc
    if not isinstance(value, dict):
        raise HeldoutProtocolError("{} must be one JSON object".format(label))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evidence-Governed Variation Evidence, Evaluation, and bounded Variation slices"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the deterministic evidence-core floor smoke")
    smoke.add_argument("--json", action="store_true", dest="as_json")
    smoke.add_argument(
        "--two-process",
        action="store_true",
        help="also run the bounded CPU-only trainer/evaluator IPC fixture",
    )

    status = subparsers.add_parser("status", help="inspect an authoritative ledger without writing it")
    status.add_argument("--ledger", required=True, type=Path)
    status.add_argument("--blob-root", type=Path)
    status.add_argument("--json", action="store_true", dest="as_json")

    export = subparsers.add_parser("export", help="write deterministic ledger JSONL")
    export.add_argument("--ledger", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--blob-root", type=Path)

    replay = subparsers.add_parser("replay", help="verify or replay deterministic ledger JSONL")
    replay.add_argument("--input", type=Path)
    replay.add_argument("--ledger", type=Path)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--blob-root", type=Path)
    replay.add_argument("--json", action="store_true", dest="as_json")

    verify = subparsers.add_parser("verify-public", help="run closed public cryptographic decision replay")
    verify.add_argument("--projection", required=True, type=Path)
    verify.add_argument("--public-key", required=True, type=Path)
    verify.add_argument("--protocol-digest", required=True)
    verify.add_argument("--evaluator-digest")
    verify.add_argument("--allow-provisional", action="store_true")
    verify.add_argument("--json", action="store_true", dest="as_json")

    evaluation = subparsers.add_parser("evaluation", help="run the bounded CPU-only Evaluation slice")
    evaluation_subparsers = evaluation.add_subparsers(dest="evaluation_command", required=True)
    evaluation_freeze = evaluation_subparsers.add_parser(
        "freeze", help="materialize deterministic trainer, public, and evaluator-private artifacts"
    )
    evaluation_freeze.add_argument("--output", required=True, type=Path)
    evaluation_freeze.add_argument("--json", action="store_true", dest="as_json")
    evaluation_smoke = evaluation_subparsers.add_parser("smoke", help="run the CPU-only Evaluation production smoke")
    evaluation_smoke.add_argument("--output", type=Path)
    evaluation_smoke.add_argument("--json", action="store_true", dest="as_json")

    variation = subparsers.add_parser("variation", help="run the bounded EGV Variation slice")
    variation_subparsers = variation.add_subparsers(dest="variation_command", required=True)
    variation_smoke = variation_subparsers.add_parser(
        "smoke", help="run the deterministic CPU-only Variation fixture smoke"
    )
    variation_smoke.add_argument("--output", type=Path)
    variation_smoke.add_argument("--json", action="store_true", dest="as_json")
    model_preflight = variation_subparsers.add_parser(
        "model-preflight", help="verify a locally staged, revision-pinned model manifest without loading it"
    )
    model_preflight.add_argument("--model-root", required=True, type=Path)
    model_preflight.add_argument("--manifest", type=Path)
    model_preflight.add_argument("--json", action="store_true", dest="as_json")
    variation_evaluator_once = variation_subparsers.add_parser(
        "evaluator-once", help="run one independent Docker-backed Variation evaluation from stdin"
    )
    variation_evaluator_once.add_argument("--service-manifest", required=True, type=Path)
    variation_evaluator_once.add_argument("--evaluator-seed", required=True, type=Path)
    variation_evaluator_once.add_argument("--private-key", required=True, type=Path)
    variation_evaluator_once.add_argument("--workspace", required=True, type=Path)
    variation_evaluator_once.add_argument("--state-root", required=True, type=Path)
    variation_service_freeze = variation_subparsers.add_parser(
        "freeze-evaluator-service", help="freeze a path-free independent evaluator service manifest"
    )
    variation_service_freeze.add_argument("--campaign-id", required=True)
    variation_service_freeze.add_argument("--model-digest", required=True)
    variation_service_freeze.add_argument("--evaluator-revision", required=True)
    variation_service_freeze.add_argument("--evaluator-seed", required=True, type=Path)
    variation_service_freeze.add_argument("--public-key", required=True, type=Path)
    variation_service_freeze.add_argument("--command", required=True, type=Path, dest="evaluator_command")
    variation_service_freeze.add_argument("--output", required=True, type=Path)

    training = subparsers.add_parser("training", help="run the bounded EGV Training runtime")
    training_subparsers = training.add_subparsers(dest="training_command", required=True)
    training_smoke = training_subparsers.add_parser(
        "smoke", help="run the CPU-only Training fixture smoke without a Qwen or promotion claim"
    )
    training_smoke.add_argument("--json", action="store_true", dest="as_json")
    evaluator_once = training_subparsers.add_parser(
        "evaluator-once", help="run one evaluator-owned signed development-loss request from stdin"
    )
    evaluator_once.add_argument("--service-manifest", required=True, type=Path)
    evaluator_once.add_argument("--model-root", required=True, type=Path)
    evaluator_once.add_argument("--development-dataset", required=True, type=Path)
    evaluator_once.add_argument("--private-key", required=True, type=Path)
    evaluator_once.add_argument("--adapter-store", required=True, type=Path)
    evaluator_once.add_argument("--device", default="cuda")
    development_freeze = training_subparsers.add_parser(
        "freeze-evaluator-service", help="freeze the exact private eight-row dev runtime and public service manifest"
    )
    development_freeze.add_argument("--campaign-id", required=True)
    development_freeze.add_argument("--model-digest", required=True)
    development_freeze.add_argument("--evaluator-seed", required=True, type=Path)
    development_freeze.add_argument("--public-key", required=True, type=Path)
    development_freeze.add_argument("--command", required=True, type=Path, dest="evaluator_command")
    development_freeze.add_argument("--transfer-command", required=True, type=Path)
    development_freeze.add_argument("--private-output", required=True, type=Path)
    development_freeze.add_argument("--service-output", required=True, type=Path)
    adapter_receive = training_subparsers.add_parser(
        "receive-adapter", help="receive a sealed adapter into evaluator-owned content-addressed storage"
    )
    adapter_receive.add_argument("--service-manifest", required=True, type=Path)
    adapter_receive.add_argument("--adapter-store", required=True, type=Path)
    adapter_receive.add_argument("--private-key", required=True, type=Path)
    training_contract = subparsers.add_parser(
        "train-lora", help="run production LoRA training from sealed local artifacts"
    )
    training_contract.add_argument("--model-root", required=True, type=Path)
    training_contract.add_argument("--train-manifest", required=True, type=Path)
    training_contract.add_argument("--development-manifest", required=True, type=Path)
    training_contract.add_argument("--evaluator-public-key", type=Path)
    training_contract.add_argument("--evaluator-command", type=Path)
    training_contract.add_argument("--evaluator-transfer-command", type=Path)
    training_contract.add_argument("--output", type=Path)
    training_contract.add_argument("--device", default="cuda")
    training_contract.add_argument("--json", action="store_true", dest="as_json")

    commissioning = subparsers.add_parser("commissioning", help="prepare and run the frozen 20/8/80 campaign")
    commissioning_subparsers = commissioning.add_subparsers(dest="commissioning_command", required=True)
    commissioning_prepare = commissioning_subparsers.add_parser(
        "prepare", help="prepare separate trainer and evaluator commissioning inputs"
    )
    commissioning_prepare.add_argument("--campaign-id", required=True)
    commissioning_prepare.add_argument("--model-digest", required=True)
    commissioning_prepare.add_argument("--model-root", required=True, type=Path)
    commissioning_prepare.add_argument("--evaluator-seed", required=True, type=Path)
    commissioning_prepare.add_argument("--trainer-output", required=True, type=Path)
    commissioning_prepare.add_argument("--trainer-sources-output", required=True, type=Path)
    commissioning_prepare.add_argument("--evaluator-output", required=True, type=Path)
    commissioning_run = commissioning_subparsers.add_parser(
        "run", help="run one frozen request or resume the complete 80-request matrix"
    )
    commissioning_run.add_argument("--trainer-inputs", required=True, type=Path)
    commissioning_run.add_argument("--trainer-sources", required=True, type=Path)
    commissioning_run.add_argument("--model-root", required=True, type=Path)
    commissioning_run.add_argument("--ledger", required=True, type=Path)
    commissioning_run.add_argument("--blob-root", required=True, type=Path)
    commissioning_run.add_argument("--evaluator-manifest", required=True, type=Path)
    commissioning_run.add_argument("--evaluator-public-key", required=True, type=Path)
    commissioning_run.add_argument("--evaluator-command", required=True, type=Path)
    commissioning_run.add_argument("--workspace", required=True, type=Path)
    commissioning_run.add_argument("--journal", required=True, type=Path)
    commissioning_run.add_argument("--source-commit", required=True)
    commissioning_run.add_argument("--request-id")
    commissioning_run.add_argument("--max-attempts", type=int, default=12)
    commissioning_run.add_argument("--device", default="cuda")
    commissioning_run.add_argument("--json", action="store_true", dest="as_json")
    commissioning_freeze = commissioning_subparsers.add_parser(
        "freeze-training", help="build the private train-lora dataset on the evaluator"
    )
    commissioning_freeze.add_argument("--trainer-inputs", required=True, type=Path)
    commissioning_freeze.add_argument("--ledger", required=True, type=Path)
    commissioning_freeze.add_argument("--blob-root", required=True, type=Path)
    commissioning_freeze.add_argument("--private-store", required=True, type=Path)
    commissioning_freeze.add_argument("--evaluator-seed", required=True, type=Path)
    commissioning_freeze.add_argument("--evaluator-public-key", required=True, type=Path)
    commissioning_freeze.add_argument("--model-root", required=True, type=Path)
    commissioning_freeze.add_argument("--output", required=True, type=Path)
    commissioning_freeze.add_argument("--json", action="store_true", dest="as_json")

    heldout = subparsers.add_parser(
        "heldout", help="freeze and run the local content-bound held-out campaign runtime"
    )
    heldout_subparsers = heldout.add_subparsers(dest="heldout_command", required=True)
    heldout_prepare = heldout_subparsers.add_parser(
        "prepare", help="freeze the held-out protocol and exact public trainer source package"
    )
    heldout_prepare.add_argument("--campaign-id", required=True)
    heldout_prepare.add_argument("--bindings", required=True, type=Path)
    heldout_prepare.add_argument("--evaluator-seed", required=True, type=Path)
    heldout_prepare.add_argument("--evaluator-public-key", required=True, type=Path)
    heldout_prepare.add_argument("--generation-profile-digest", required=True)
    heldout_prepare.add_argument("--schedule-seed", required=True, type=int)
    heldout_prepare.add_argument("--bootstrap-seed", required=True, type=int)
    heldout_prepare.add_argument("--protocol-output", required=True, type=Path)
    heldout_prepare.add_argument("--trainer-inputs-output", required=True, type=Path)
    heldout_prepare.add_argument("--trainer-sources-output", required=True, type=Path)
    heldout_prepare.add_argument("--json", action="store_true", dest="as_json")
    heldout_service = heldout_subparsers.add_parser(
        "freeze-evaluator-service", help="freeze the path-free evaluator identity manifest"
    )
    heldout_service.add_argument("--protocol", required=True, type=Path)
    heldout_service.add_argument("--output", required=True, type=Path)
    heldout_service.add_argument("--json", action="store_true", dest="as_json")
    heldout_evaluator = heldout_subparsers.add_parser(
        "evaluator-once", help="process one local content-bound evaluator command from stdin"
    )
    heldout_evaluator.add_argument("--protocol", required=True, type=Path)
    heldout_evaluator.add_argument("--service-manifest", required=True, type=Path)
    heldout_evaluator.add_argument("--private-key", required=True, type=Path)
    heldout_evaluator.add_argument("--state-root", required=True, type=Path)
    heldout_evaluator.add_argument("--observation-verifier", required=True)
    heldout_evaluator.add_argument("--reconciliation-probe")
    heldout_run = heldout_subparsers.add_parser(
        "run", help="resume the frozen schedule through explicitly supplied local runtime hooks"
    )
    heldout_run.add_argument("--protocol", required=True, type=Path)
    heldout_run.add_argument("--journal", required=True, type=Path)
    heldout_run.add_argument("--operations", required=True, type=Path)
    heldout_run.add_argument("--runner", required=True)
    heldout_run.add_argument("--result-verifier", required=True)
    heldout_run.add_argument("--reconciler", required=True)
    heldout_run.add_argument("--continue-after-integrity-failure", action="store_true")
    heldout_run.add_argument("--json", action="store_true", dest="as_json")
    heldout_finalize = heldout_subparsers.add_parser(
        "finalize", help="build a private final record from all signed outcomes and restoration evidence"
    )
    heldout_finalize.add_argument("--protocol", required=True, type=Path)
    heldout_finalize.add_argument("--journal", required=True, type=Path)
    heldout_finalize.add_argument("--restoration-receipt", required=True, type=Path)
    heldout_finalize.add_argument("--private-output", required=True, type=Path)
    heldout_finalize.add_argument("--json", action="store_true", dest="as_json")

    for phase in sorted(SLICE2_PHASES):
        phase_parser = subparsers.add_parser(phase, help=f"{phase} (outside Slice 2; fails closed)")
        phase_parser.add_argument("--campaign-id")
        phase_parser.add_argument("--inventory-reference")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "smoke":
            result = run_core_smoke()
            if args.two_process:
                result["two_process"] = run_two_process_smoke()
            _json_output(result, args.as_json)
            return 0
        if args.command == "status":
            with EvidenceLedger(args.ledger, mode="read_only", blob_root=args.blob_root) as ledger:
                _json_output(ledger.status(), args.as_json)
            return 0
        if args.command == "export":
            with EvidenceLedger(args.ledger, mode="read_only", blob_root=args.blob_root) as ledger:
                ledger.export_jsonl(args.output)
            print(str(args.output))
            return 0
        if args.command == "replay":
            if args.ledger is not None and args.input is None:
                with EvidenceLedger(args.ledger, mode="read_only", blob_root=args.blob_root) as ledger:
                    result = {"integrity": ledger.verify_integrity(), "status": ledger.status()}
                _json_output(result, args.as_json)
                return 0
            if args.input is None or args.output is None:
                raise EGVError("replay requires --input and --output, or --ledger for read-only verification")
            replayed = EvidenceLedger.replay_jsonl(args.input, args.output, blob_root=args.blob_root)
            try:
                result = {"replayed_to": str(args.output), "integrity": replayed.verify_integrity()}
            finally:
                replayed.close()
            _json_output(result, args.as_json)
            return 0
        if args.command == "verify-public":
            projection = load_public_projection(args.projection)
            verifier = PublicCryptographicVerifier(
                args.public_key.read_bytes(),
                protocol_digest=args.protocol_digest,
                evaluator_digest=args.evaluator_digest,
                require_terminal_seal=not args.allow_provisional,
            )
            report = verifier.verify(
                candidates=projection.candidates.values(),
                dependencies=projection.dependencies.values(),
                receipts=projection.receipts.values(),
                events=projection.events,
                restore_receipt=projection.restore_receipt,
            )
            _json_output(report.to_dict(), args.as_json)
            return 0
        if args.command == "evaluation":
            if args.evaluation_command == "freeze":
                report = freeze_evaluation(args.output)
                _json_output(report.to_dict(), args.as_json)
                return 0
            if args.evaluation_command == "smoke":
                _json_output(run_evaluation_smoke(args.output), args.as_json)
                return 0
            raise EGVError("unsupported evaluation command: {}".format(args.evaluation_command))
        if args.command == "variation":
            if args.variation_command == "smoke":
                _json_output(run_variation_smoke(args.output), args.as_json)
                return 0
            if args.variation_command == "model-preflight":
                loader = PinnedModelLoader(args.model_root, manifest_path=args.manifest)
                _json_output(loader.preflight(), args.as_json)
                return 0
            if args.variation_command == "evaluator-once":
                stdin_stream = getattr(sys.stdin, "buffer", sys.stdin)
                raw_request = stdin_stream.read(REMOTE_VARIATION_REQUEST_LIMIT + 1)
                if isinstance(raw_request, str):
                    raw_request = raw_request.encode("utf-8")
                if len(raw_request) > REMOTE_VARIATION_REQUEST_LIMIT:
                    raise EGVError("remote Variation evaluator request exceeds the bounded input limit")
                try:
                    request = json.loads(raw_request.decode("utf-8"))
                except (UnicodeError, ValueError) as exc:
                    raise EGVError("remote Variation evaluator request is not valid JSON") from exc
                response = run_remote_evaluator_once(
                    request,
                    service_manifest=args.service_manifest,
                    evaluator_seed=args.evaluator_seed,
                    evaluator_private_key=args.private_key,
                    workspace=args.workspace,
                    state_root=args.state_root,
                )
                print(canonical_json(response))
                return 0
            if args.variation_command == "freeze-evaluator-service":
                from .evaluation.authority import AuthorityPolicy
                from .evaluation.sandbox import DockerSandboxConfig

                corpus = EvaluationCorpus.generate(secret_seed_file=args.evaluator_seed)
                manifest = build_remote_evaluator_service_manifest(
                    campaign_id=args.campaign_id,
                    model_digest=args.model_digest,
                    protocol_digest=VARIATION_PROTOCOL_DIGEST,
                    policy_digest=AuthorityPolicy.candidate_execution().digest,
                    corpus=corpus,
                    evaluator_revision=args.evaluator_revision,
                    public_key_path=args.public_key,
                    command=args.evaluator_command,
                    docker_config=DockerSandboxConfig.from_environment(),
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
                _json_output(
                    {
                        "service_manifest": str(args.output),
                        "service_manifest_digest": manifest["service_manifest_digest"],
                        "task_count": len(manifest["task_bindings"]),
                    },
                    True,
                )
                return 0
            raise EGVError("unsupported variation command: {}".format(args.variation_command))
        if args.command == "training":
            if args.training_command == "smoke":
                _json_output(run_training_smoke(), args.as_json)
                return 0
            if args.training_command == "evaluator-once":
                try:
                    request = json.loads(sys.stdin.read())
                except ValueError as exc:
                    raise EGVError("external evaluator request is not valid JSON") from exc
                response = run_external_evaluator_once(
                    request,
                    service_manifest=args.service_manifest,
                    model_root=args.model_root,
                    development_dataset=args.development_dataset,
                    evaluator_private_key=args.private_key,
                    adapter_store=args.adapter_store,
                    device=args.device,
                )
                print(canonical_json(response))
                return 0
            if args.training_command == "freeze-evaluator-service":
                from .training import TrainingProtocol

                corpus = EvaluationCorpus.generate(secret_seed_file=args.evaluator_seed)
                service = freeze_external_development_service(
                    corpus=corpus,
                    campaign_id=args.campaign_id,
                    model_digest=args.model_digest,
                    protocol_digest=TrainingProtocol().digest,
                    public_key_path=args.public_key,
                    command=args.evaluator_command,
                    transfer_command=args.transfer_command,
                    private_output=args.private_output,
                    service_output=args.service_output,
                )
                _json_output({
                    "private_runtime": str(args.private_output),
                    "service_manifest": str(args.service_output),
                    "service_manifest_digest": service["service_manifest_digest"],
                    "development_task_count": 8,
                }, True)
                return 0
            if args.training_command == "receive-adapter":
                try:
                    request = json.loads(sys.stdin.read())
                except ValueError as exc:
                    raise EGVError("external adapter transfer request is not valid JSON") from exc
                response = receive_external_adapter(
                    request,
                    service_manifest=args.service_manifest,
                    adapter_store=args.adapter_store,
                    evaluator_private_key=args.private_key,
                )
                print(canonical_json(response))
                return 0
            raise EGVError("unsupported training command: {}".format(args.training_command))
        if args.command == "commissioning":
            if args.commissioning_command == "prepare":
                plan = prepare_commissioning(
                    EvaluationCorpus.generate(secret_seed_file=args.evaluator_seed),
                    campaign_id=args.campaign_id,
                    model_manifest_digest=args.model_digest,
                    generation_profile_digest=_commissioning_generation_profile(
                        args.model_root, args.model_digest
                    ),
                )
                trainer_output, trainer_sources_output, evaluator_output = _commissioning_output_targets(
                    args.trainer_output, args.trainer_sources_output, args.evaluator_output
                )
                _atomic_publish_new_json(trainer_output, plan.trainer_inputs())
                try:
                    _atomic_publish_new_json(trainer_sources_output, plan.trainer_source_manifest)
                    _atomic_publish_new_json(evaluator_output, plan.private_manifest())
                except Exception:
                    trainer_output.unlink(missing_ok=True)
                    trainer_sources_output.unlink(missing_ok=True)
                    evaluator_output.unlink(missing_ok=True)
                    raise
                _json_output({
                    "trainer_inputs_digest": plan.trainer_inputs()["trainer_inputs_digest"],
                    "trainer_sources_digest": plan.trainer_sources_digest,
                    "private_inputs_digest": digest_for(plan.private_manifest()),
                    "generation_profile_digest": plan.generation_profile_digest,
                    "request_count": len(plan.generation_requests),
                    "live_model_executed": False,
                }, True)
                return 0
            if args.commissioning_command == "run":
                result = run_commissioning(
                    trainer_inputs_path=args.trainer_inputs,
                    trainer_sources_path=args.trainer_sources, model_root=args.model_root,
                    ledger_path=args.ledger, blob_root=args.blob_root,
                    evaluator_manifest=args.evaluator_manifest,
                    evaluator_public_key=args.evaluator_public_key,
                    evaluator_command=args.evaluator_command, workspace_root=args.workspace,
                    journal_path=args.journal, source_commit=args.source_commit,
                    request_id=args.request_id, max_attempts=args.max_attempts, device=args.device,
                )
                _json_output(result, args.as_json)
                return 0
            if args.commissioning_command == "freeze-training":
                result = freeze_commissioning_dataset(
                    trainer_inputs_path=args.trainer_inputs, ledger_path=args.ledger,
                    blob_root=args.blob_root, private_store_root=args.private_store,
                    evaluator_seed=args.evaluator_seed,
                    evaluator_public_key=args.evaluator_public_key,
                    model_root=args.model_root, output=args.output,
                )
                _json_output(result, args.as_json)
                return 0
            raise EGVError("unsupported commissioning command: {}".format(args.commissioning_command))
        if args.command == "heldout":
            if args.heldout_command == "prepare":
                outputs = _commissioning_output_targets(
                    args.protocol_output, args.trainer_inputs_output, args.trainer_sources_output
                )
                bindings = _load_canonical_json(args.bindings, "held-out bindings", limit=65_536)
                public_key = _read_ed25519_key(
                    args.evaluator_public_key, "held-out evaluator public key", private=False
                )
                evaluator_seed = _read_regular_bytes(
                    args.evaluator_seed,
                    "held-out evaluator seed",
                    limit=32,
                    exact_size=32,
                    private=True,
                )
                corpus = EvaluationCorpus.generate(secret_seed_file=args.evaluator_seed)
                if corpus.private_seed_bytes() != evaluator_seed:
                    raise HeldoutProtocolError("held-out evaluator seed changed during corpus construction")
                protocol = FrozenHeldoutProtocol.build(
                    campaign_id=args.campaign_id,
                    bindings=bindings,
                    evaluator_public_key=public_key,
                    schedule_seed=args.schedule_seed,
                    bootstrap_seed=args.bootstrap_seed,
                    heldout_task_records=ordered_public_heldout_task_records(corpus),
                )
                trainer_inputs, trainer_sources = build_trainer_evidence_package(
                    corpus,
                    protocol,
                    generation_profile_digest=args.generation_profile_digest,
                )
                _publish_json_transaction((
                    (outputs[0], protocol.to_private_dict()),
                    (outputs[1], trainer_inputs),
                    (outputs[2], trainer_sources),
                ))
                _json_output(
                    {
                        "protocol_digest": protocol.digest,
                        "coordinates": len(protocol.coordinates),
                        "trainer_inputs_digest": trainer_inputs["trainer_inputs_digest"],
                        "trainer_sources_digest": trainer_sources["trainer_sources_digest"],
                    },
                    args.as_json,
                )
                return 0
            if args.heldout_command == "freeze-evaluator-service":
                protocol = _load_heldout_protocol(args.protocol)
                manifest = HeldoutVerifierServiceManifest.from_protocol(protocol)
                _atomic_publish_new_json(args.output, manifest.to_dict())
                _json_output(
                    {"service_manifest": str(args.output), "service_manifest_digest": manifest.digest},
                    args.as_json,
                )
                return 0
            if args.heldout_command == "evaluator-once":
                protocol = _load_heldout_protocol(args.protocol)
                manifest = HeldoutVerifierServiceManifest.load(
                    _load_canonical_json(
                        args.service_manifest,
                        "held-out evaluator service manifest",
                        limit=65_536,
                    ),
                    protocol,
                )
                signer = ReceiptSigner(_read_ed25519_key(
                    args.private_key, "held-out evaluator private key", private=True
                ))
                if signer.key_id != protocol.evaluator_key_id:
                    raise HeldoutProtocolError("held-out private key differs from the frozen evaluator")
                observation_verifier = _approved_integration(
                    HELDOUT_OBSERVATION_VERIFIERS,
                    args.observation_verifier,
                    "held-out observation verifier",
                )
                reconciliation_probe = (
                    _approved_integration(
                        HELDOUT_RECONCILIATION_PROBES,
                        args.reconciliation_probe,
                        "held-out reconciliation probe",
                    )
                    if args.reconciliation_probe is not None
                    else None
                )
                try:
                    response = run_heldout_verifier_once(
                        protocol,
                        manifest,
                        _read_bounded_stdin_json("held-out evaluator request"),
                        signer=signer,
                        state_root=args.state_root,
                        observation_verifier=observation_verifier,
                        reconciliation_probe=reconciliation_probe,
                    )
                except (ImportError, RuntimeError):
                    raise HeldoutProtocolError("held-out evaluator integration failed safely") from None
                print(canonical_json(response))
                return 0
            if args.heldout_command == "run":
                protocol = _load_heldout_protocol(args.protocol)
                journal = HeldoutJournal(args.journal, protocol)
                operations = CoordinateOperationStore(args.operations, protocol)
                runner = _approved_integration(
                    HELDOUT_COORDINATE_RUNNERS, args.runner, "held-out coordinate runner"
                )
                result_verifier = _approved_integration(
                    HELDOUT_RESULT_VERIFIERS, args.result_verifier, "held-out result verifier"
                )
                reconciler = _approved_integration(
                    HELDOUT_RECONCILERS, args.reconciler, "held-out reconciler"
                )
                try:
                    result = run_pending_coordinates(
                        protocol,
                        journal,
                        operations,
                        runner,
                        result_verifier,
                        reconciler,
                        halt_on_integrity_failure=not args.continue_after_integrity_failure,
                    )
                except (ImportError, RuntimeError):
                    raise HeldoutProtocolError("held-out campaign integration failed safely") from None
                _json_output(result, args.as_json)
                return 0
            if args.heldout_command == "finalize":
                protocol = _load_heldout_protocol(args.protocol)
                journal = HeldoutJournal(args.journal, protocol)
                restoration = _load_canonical_json(
                    args.restoration_receipt, "held-out restoration receipt", limit=262_144
                )
                private_record = build_private_campaign_record(
                    protocol, journal.envelopes, restoration
                )
                _atomic_publish_new_json(args.private_output, private_record)
                analysis = private_record["aggregate_analysis"]
                _json_output(
                    {
                        "private_output": str(args.private_output),
                        "protocol_digest": protocol.digest,
                        "signed_outcomes": len(journal.envelopes),
                        "disposition": analysis["DISPOSITION"],
                        "gate_vector": analysis["GATE_VECTOR"],
                    },
                    args.as_json,
                )
                return 0
            raise EGVError("unsupported heldout command: {}".format(args.heldout_command))
        if args.command == "train-lora":
            if (
                args.evaluator_public_key is None or args.evaluator_command is None
                or args.evaluator_transfer_command is None or args.output is None
            ):
                raise PhaseUnavailable(
                    "train-lora must fail closed without external evaluator authority: provide its frozen public key, "
                    "content-bound command, service manifest, and an output directory"
                )
            result = run_production_training(
                model_root=args.model_root,
                training_dataset=args.train_manifest,
                evaluator_manifest=args.development_manifest,
                evaluator_public_key=args.evaluator_public_key,
                evaluator_command=args.evaluator_command,
                evaluator_transfer_command=args.evaluator_transfer_command,
                output_root=args.output,
                device=args.device,
            )
            _json_output(result, args.as_json)
            return 0
        if args.command in SLICE2_PHASES:
            raise PhaseUnavailable(
                f"{args.command} is outside Slice 2 evidence core; no services, credentials, hidden tests, "
                "private restore inventory, active DeepSeek workload, or hosted model was accessed"
            )
        raise EGVError(f"unsupported command: {args.command}")
    except (CampaignError, EGVError, OSError, ValueError) as exc:
        print(f"egv: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main", "run_core_smoke"]
