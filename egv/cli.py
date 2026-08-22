"""Safe Slice 2 command-line paths for evidence storage and replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from .canonical import canonical_json, digest_for
from .campaign.commissioning import prepare_commissioning
from .campaign.errors import CampaignError
from .campaign.runner import freeze_commissioning_dataset, run_commissioning
from .evaluation.artifacts import freeze_evaluation
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
from .training import run_external_evaluator_once, run_production_training, run_training_smoke
from .variation import (
    PinnedModelLoader,
    VARIATION_PROTOCOL_DIGEST,
    build_remote_evaluator_service_manifest,
    run_remote_evaluator_once,
    run_variation_smoke,
)
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
                "adapter_digest": _sha("egv-smoke-adapter"),
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
    variation_service_freeze.add_argument("--command", required=True, type=Path)
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
    evaluator_once.add_argument("--device", default="cuda")
    training_contract = subparsers.add_parser(
        "train-lora", help="run production LoRA training from sealed local artifacts"
    )
    training_contract.add_argument("--model-root", required=True, type=Path)
    training_contract.add_argument("--train-manifest", required=True, type=Path)
    training_contract.add_argument("--development-manifest", required=True, type=Path)
    training_contract.add_argument("--evaluator-public-key", type=Path)
    training_contract.add_argument("--evaluator-command", type=Path)
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
    commissioning_prepare.add_argument("--evaluator-seed", required=True, type=Path)
    commissioning_prepare.add_argument("--trainer-output", required=True, type=Path)
    commissioning_prepare.add_argument("--evaluator-output", required=True, type=Path)
    commissioning_run = commissioning_subparsers.add_parser(
        "run", help="run one frozen request or resume the complete 80-request matrix"
    )
    commissioning_run.add_argument("--trainer-inputs", required=True, type=Path)
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
                from .evaluation.dataset import EvaluationCorpus
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
                    command=args.command,
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
                    device=args.device,
                )
                print(canonical_json(response))
                return 0
            raise EGVError("unsupported training command: {}".format(args.training_command))
        if args.command == "commissioning":
            if args.commissioning_command == "prepare":
                from .evaluation.dataset import EvaluationCorpus

                plan = prepare_commissioning(
                    EvaluationCorpus.generate(secret_seed_file=args.evaluator_seed),
                    campaign_id=args.campaign_id,
                    model_manifest_digest=args.model_digest,
                )
                args.trainer_output.parent.mkdir(parents=True, exist_ok=True)
                args.evaluator_output.parent.mkdir(parents=True, exist_ok=True)
                args.trainer_output.write_text(canonical_json(plan.trainer_inputs()) + "\n", encoding="utf-8")
                args.evaluator_output.write_text(canonical_json(plan.private_manifest()) + "\n", encoding="utf-8")
                _json_output({
                    "trainer_inputs_digest": plan.trainer_inputs()["trainer_inputs_digest"],
                    "private_inputs_digest": digest_for(plan.private_manifest()),
                    "request_count": len(plan.generation_requests),
                    "live_model_executed": False,
                }, True)
                return 0
            if args.commissioning_command == "run":
                result = run_commissioning(
                    trainer_inputs_path=args.trainer_inputs, model_root=args.model_root,
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
        if args.command == "train-lora":
            if args.evaluator_public_key is None or args.evaluator_command is None or args.output is None:
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
