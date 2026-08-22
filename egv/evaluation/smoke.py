"""CPU-only production smoke for the complete Evaluation slice."""

from __future__ import annotations

from pathlib import Path
import secrets
import tempfile
from typing import Any, Dict, Optional, Union

from .artifacts import assert_byte_identical_regeneration, freeze_evaluation, scan_public_artifacts
from .authority import DockerEnforcedRuntime, OpenShellRuntime
from .dataset import EvaluationCorpus
from .errors import EGVError
from .shock import CorrectionShockSuite
from .process_smoke import run_evaluation_two_process_smoke


def run_evaluation_smoke(output_root: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Exercise corpus freeze, sandbox controls, hidden comparison, receipts, and shock audit."""

    temporary = None
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="egv-evaluation-smoke-")
        root = Path(temporary.name)
    else:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
    seed_path = root / "evaluator-private" / "corpus-seed.bin"
    if not seed_path.exists():
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_bytes(secrets.token_bytes(32))
        seed_path.chmod(0o600)
    corpus = EvaluationCorpus.generate(secret_seed_file=seed_path)
    try:
        freeze_a = root / "freeze-a"
        freeze_b = root / "freeze-b"
        report_a = freeze_evaluation(freeze_a, corpus=corpus)
        freeze_evaluation(freeze_b, corpus=corpus)
        assert_byte_identical_regeneration(freeze_a, freeze_b)
        public_scan = scan_public_artifacts(freeze_a, corpus)

        runtime = DockerEnforcedRuntime()
        shock = CorrectionShockSuite().audit_dependency_aware()
        two_process = run_evaluation_two_process_smoke(
            corpus=corpus,
            evaluator_seed_file=freeze_a / "evaluator-private" / "corpus-seed.bin",
            frozen_root=freeze_a,
            frozen_data_manifest_digest=report_a.data_manifest_digest,
        )
        if two_process["evaluated_corpus_digest"] != report_a.data_manifest_digest:
            raise EGVError("evaluation smoke froze and evaluated different corpus manifests")
        if not all(
            binding["source_bytes_match_frozen"]
            and binding["evaluated_source_digest"] == binding["frozen_candidate_source_digest"]
            for binding in two_process["evaluated_source_bindings"].values()
        ):
            raise EGVError("evaluation smoke evaluated source bytes different from frozen held-out artifacts")
        evaluation = dict(two_process["family_results"]["PURE_FUNCTION"])
        evaluation_receipt_count = len(evaluation.get("receipt_ids", ()))
        evaluation.pop("receipt_ids", None)
        evaluation["receipt_count"] = evaluation_receipt_count
        post_integrity = two_process["post_integrity"]
        post_status = two_process["post_status"]
        journal = two_process["journal"]
        return {
            "smoke": "PASS",
            "runtime_tier": "ceiling-docker-evaluation-fixture",
            "model_dependencies": False,
            "openshell": OpenShellRuntime().status.to_dict(),
            "authority": runtime.status.to_dict(),
            "corpus": {
                "task_count": 36,
                "split_counts": {"train": 20, "dev": 8, "heldout": 8},
                "byte_identical_regeneration": True,
                "data_manifest_digest": report_a.data_manifest_digest,
                "evaluated_data_manifest_digest": two_process["evaluated_corpus_digest"],
                "evaluated_source_bindings": two_process["evaluated_source_bindings"],
            },
            "freeze": {
                "protocol_digest": report_a.protocol_digest,
                "file_count": report_a.file_count,
                "public_scan": public_scan,
            },
            "determinism": {
                "non_key_artifacts_byte_identical": True,
                "key_material_excluded": True,
            },
            "sandbox_negative_controls": two_process["negative_controls"],
            "evaluation": evaluation,
            "receipts": {
                "journal": {"count": journal["count"], "chain_valid": journal["chain_valid"]},
                "ledger": {"receipt_count": post_integrity["receipt_count"], "chain_valid": post_integrity["chain_valid"]},
                "reconciled_again": 0,
                "ledger_status": {
                    "receipt_count": post_status["receipt_count"],
                    "quarantine_count": post_status["quarantine_count"],
                    "quarantined": post_status["quarantined"],
                },
                "candidate_disposition": two_process["post_disposition"],
                "integrity": dict(post_integrity),
            },
            "correction_shock": shock,
            "two_process": two_process,
            "limitations": [
                "no Qwen, GPU, hosted model, network, DeepSeek, or OpenShell claim",
                "Docker production smoke is a bounded fixture, not a dual-Spark campaign",
                "hidden expected outputs remain evaluator-private",
                "the pinned Docker image must already be cached; image pull is forbidden",
            ],
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


__all__ = ["run_evaluation_smoke"]
