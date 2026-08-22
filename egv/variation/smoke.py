"""Bounded CPU-only Variation smoke using private synthetic fixture inputs."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
from typing import Any, Dict, Optional, Union

from ..canonical import digest_for
from ..evaluation.dataset import EVALUATOR_SEED_BYTES, EvaluationCorpus
from ..ledger import EvidenceLedger
from .arms import ArmIsolation
from .fixture import FixtureEvaluationGateway
from .generator import DeterministicFixtureGenerator
from .loop import BoundedCandidateLoop, VariationTask
from .model import MODEL_REVISION


def _write_private_seed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"V" * EVALUATOR_SEED_BYTES)
    path.chmod(0o600)


def run_variation_smoke(output_root: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Exercise the ledger, retrieval, candidate loop, and checkpoint path.

    This is an explicitly named fixture tier.  It does not instantiate Qwen,
    Docker, a GPU, a LoRA adapter, or an authority runtime.  It is useful for
    CPU wiring and replay evidence; production runs reject this evaluator.
    """

    temporary: Optional[Any] = None
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="egv-variation-smoke-")
        root = Path(temporary.name)
    else:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
    try:
        private_root = root / "private" / "evaluator-private"
        seed_path = private_root / "corpus-seed.bin"
        _write_private_seed(seed_path)
        corpus = EvaluationCorpus.generate(secret_seed_file=seed_path)
        repo = corpus.hidden_repositories()[0]
        task = VariationTask.from_microrepo(repo)
        public_source = dict(repo.source_files)["src/task.py"]
        model_digest = digest_for({"fixture_model": MODEL_REVISION, "tier": "fixture"})
        policy_digest = digest_for("egv-variation-fixture-policy-v1")
        campaign_id = "variation-smoke-campaign"
        ledger = EvidenceLedger(
            root / "private" / "ledger.sqlite",
            blob_root=root / "private" / "ledger-private",
            clock=lambda: "2026-08-22T00:00:00Z",
        )
        try:
            evaluator = FixtureEvaluationGateway(
                corpus,
                ledger,
                root / "private" / "evaluator-fixture",
                policy_digest=policy_digest,
                campaign_id=campaign_id,
            )
            isolation = ArmIsolation(root / "private" / "arm-state", campaign_id=campaign_id)
            generator = DeterministicFixtureGenerator(
                {task.task_id: (public_source, repo.corrected_source)},
                public_locus={task.task_id: task.public_locus},
                model_digest=model_digest,
            )
            runner = BoundedCandidateLoop(
                ledger=ledger,
                evaluator=evaluator,
                generator=generator,
                isolation=isolation,
                workspace_root=root / "private" / "variation-state",
                campaign_id=campaign_id,
                source_commit="variation-smoke-fixture",
                model_revision=MODEL_REVISION,
                model_digest=model_digest,
                data_manifest_digest=corpus.manifest_digest(),
                policy_digest=policy_digest,
                arm_id="C",
                max_attempts=2,
                seed_set=(0, 1, 2),
                fixture_mode=True,
            )
            report = runner.run(task, seed=0)
            if not report.promoted or len(report.attempts) != 2:
                raise RuntimeError("Variation fixture did not exercise failure retrieval before promotion")
            if report.attempts[0].diagnostic_enum != "WRONG_OUTPUT":
                raise RuntimeError("Variation fixture first attempt did not produce a bounded failure")
            if report.attempts[1].disposition != "PROMOTED":
                raise RuntimeError("Variation fixture second attempt did not promote")
            if ledger.candidate_disposition(report.attempts[0].candidate_id) != "REJECTED":
                raise RuntimeError("Variation fixture rejected candidate is not retrievable as failure evidence")
            if not (root / "private" / "arm-state" / "arms" / "C" / "runs" / report.run_id).is_dir():
                raise RuntimeError("Variation arm workspace was not isolated and checkpointed")
            public_report = report.to_public_dict()
            serialized = json.dumps(public_report, sort_keys=True)
            if "corpus-seed.bin" in serialized or seed_path.as_posix() in serialized:
                raise RuntimeError("Variation smoke report leaked evaluator-private seed material")
            result = {
                "smoke": "PASS",
                "runtime_tier": "floor-fixture",
                "fixture_only": True,
                "campaign_path_exercised": True,
                "model_loader_exercised": False,
                "model": {
                    "repository": "Qwen/Qwen3.5-2B-Base",
                    "revision": MODEL_REVISION,
                    "local_model_loaded": False,
                },
                "authority": {
                    "enforceable": False,
                    "backend": "fixture-only-local-helper",
                    "production_requires": "Docker-backed ControllerEvaluationGateway",
                },
                "retrieval": {
                    "policy": report.retrieval_policy,
                    "failure_then_success": [
                        report.attempts[0].diagnostic_enum,
                        report.attempts[1].disposition,
                    ],
                    "failure_evidence_retrieved_on_attempt_2": bool(report.attempts[1].evidence_ids),
                },
                "variation": public_report,
                "split_secrecy": {
                    "seed_published": False,
                    "oracle_value_published": False,
                    "candidate_source_published": False,
                    "private_seed_file_mode": "0600",
                },
                "limitations": [
                    "fixture evaluator uses the test-only local helper and does not prove Docker authority",
                    "pinned Qwen loader preflight/load is a separate model-root command and requires local Transformers >=5.5,<6",
                    "LoRA arms E-H fail closed until the later Training slice supplies a sealed adapter",
                    "no GPU, network, DeepSeek, hosted model, or Campaign phase was accessed",
                ],
            }
            if output_root is not None:
                output_path = root / "public" / "variation-smoke-report.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
                output_path.chmod(0o444)
                result["artifact"] = output_path.name
            return result
        finally:
            ledger.close()
    finally:
        if temporary is not None:
            temporary.cleanup()


__all__ = ["run_variation_smoke"]
