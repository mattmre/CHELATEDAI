"""Content-addressed Evaluation artifacts and split-leakage scanning."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import secrets
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from ..canonical import canonical_json, digest_bytes, digest_for
from .dataset import EVALUATOR_SEED_BYTES, EvaluationCorpus, PUBLIC_HELDOUT_TEMPLATE_IDS
from .errors import ArtifactError, LeakageError
from .prompts import PromptRegistry


@dataclass(frozen=True)
class ArtifactRef:
    digest: str
    size: int
    relative_path: str
    media_type: str
    role: str


class ContentAddressedArtifactStore:
    """Write-once SHA-256 artifacts with no absolute paths in content."""

    def __init__(self, root: Union[Path, str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, *, media_type: str, role: str) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise ArtifactError("artifact bytes must be bytes")
        digest = digest_bytes(data)
        relative = Path("blobs") / "sha256" / digest[:2] / digest[2:4] / digest
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ArtifactError("content-addressed artifact path already contains different bytes")
        if not target.exists():
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(data)
            os.replace(str(temporary), str(target))
            target.chmod(0o444)
        return ArtifactRef(digest, len(data), relative.as_posix(), media_type, role)

    def read(self, digest: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ArtifactError("artifact digest must be a lowercase SHA-256 value")
        path = self.root / "blobs" / "sha256" / digest[:2] / digest[2:4] / digest
        if not path.exists():
            raise ArtifactError("content-addressed artifact is missing")
        data = path.read_bytes()
        if digest_bytes(data) != digest:
            raise ArtifactError("content-addressed artifact digest mismatch")
        return data


def _write_bytes(path: Path, data: bytes, *, mode: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ArtifactError("frozen artifact path already contains different bytes")
    path.write_bytes(data)
    if mode is not None:
        path.chmod(mode)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def _files(root: Path) -> Tuple[Path, ...]:
    return tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def _same_tree(first: Path, second: Path) -> bool:
    first_files = [path.relative_to(first).as_posix() for path in _files(first)]
    second_files = [path.relative_to(second).as_posix() for path in _files(second)]
    if first_files != second_files:
        return False
    return all((first / relative).read_bytes() == (second / relative).read_bytes() for relative in first_files)


def _publishable_files(root: Path) -> Tuple[Path, ...]:
    files = [path for area in ("frozen", "trainer", "public") for path in _files(root / area) if path.is_file()]
    manifest = root / "artifact-manifest.json"
    if manifest.exists():
        files.append(manifest)
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def scan_public_artifacts(root: Union[Path, str], corpus: Optional[EvaluationCorpus] = None) -> Dict[str, Any]:
    """Scan every publishable root after final manifests have been written."""

    root_path = Path(root)
    # The held-out identifier vocabulary is public and frozen by the ADR.  The
    # evaluator corpus adds the same IDs but is not required to make the scan
    # complete; a caller can never turn this into an unchecked pass by omitting
    # private corpus material.
    heldout_ids = set(PUBLIC_HELDOUT_TEMPLATE_IDS)
    forbidden = (
        re.compile(r"BEGIN (?:RSA|OPENSSH|EC|ED25519) PRIVATE KEY"),
        re.compile(r"(?i)(?:password|api[_-]?key|access[_-]?token|secret)\s*[:=]"),
        re.compile(r"(?i)(?:/home/|/root/|/Users/|[A-Z]:\\)"),
        re.compile(r"(?i)(?:/tmp/|/var/|/etc/|https?://|ssh://|file://)"),
        re.compile(r"(?i)(?:localhost|127\.0\.0\.1|0\.0\.0\.0|::1)"),
        re.compile(r"(?i)(?:evaluator[-_]private|golden_patch|expected_output|hidden_rule_id|hidden_test)"),
        re.compile(r"(?i)(?:corpus[-_]seed|private[-_]seed|ledger\.sqlite|docker\.sock)"),
    )
    findings: List[str] = []
    scanned = 0
    for area in ("frozen", "trainer", "public"):
        area_path = root_path / area
        if not area_path.exists():
            findings.append("missing publishable artifact area: {}".format(area))
            continue
        for path in _files(area_path):
            scanned += 1
            relative = path.relative_to(root_path).as_posix()
            text = path.read_bytes().decode("utf-8", errors="replace")
            for template_id in sorted(heldout_ids):
                if template_id in text or template_id in relative:
                    findings.append("held-out template ID leaked into {}".format(relative))
            for pattern in forbidden:
                if pattern.search(text) or pattern.search(relative):
                    findings.append("prohibited public material matched in {}".format(relative))
    manifest = root_path / "artifact-manifest.json"
    if manifest.exists():
        scanned += 1
        text = manifest.read_bytes().decode("utf-8", errors="replace")
        for template_id in sorted(heldout_ids):
            if template_id in text:
                findings.append("held-out template ID leaked into artifact-manifest.json")
        for pattern in forbidden:
            if pattern.search(text):
                findings.append("prohibited public material matched in artifact-manifest.json")
    if findings:
        raise LeakageError("; ".join(sorted(set(findings))))
    return {
        "scanned_files": scanned,
        "heldout_ids_absent": True,
        "heldout_ids_checked": True,
        "heldout_ids_status": "CHECKED_PUBLIC_CONTRACT" if corpus is None else "CHECKED",
        "scanned_views": ["frozen", "trainer", "public", "artifact-manifest.json"],
        "findings": [],
    }


@dataclass(frozen=True)
class FreezeReport:
    data_manifest_digest: str
    protocol_digest: str
    public_scan: Mapping[str, Any]
    file_count: int
    public_file_count: int
    private_file_count: int
    split_counts: Mapping[str, int]

    def to_dict(self) -> Dict[str, Any]:
        # Never return the operator's output path or any private artifact path.
        return {
            "data_manifest_digest": self.data_manifest_digest,
            "protocol_digest": self.protocol_digest,
            "public_scan": dict(self.public_scan),
            "file_count": self.file_count,
            "public_file_count": self.public_file_count,
            "private_file_count": self.private_file_count,
            "split_counts": dict(self.split_counts),
            "layout": ["frozen", "trainer", "public", "evaluator-private"],
        }


def _write_seed(root: Path, corpus: Optional[EvaluationCorpus]) -> EvaluationCorpus:
    seed_path = root / "evaluator-private" / "corpus-seed.bin"
    if corpus is None:
        _write_bytes(seed_path, secrets.token_bytes(EVALUATOR_SEED_BYTES), mode=0o600)
        return EvaluationCorpus.generate(secret_seed_file=seed_path)
    corpus.validate()
    _write_bytes(seed_path, corpus.private_seed_bytes(), mode=0o600)
    return corpus


def _private_manifest(root: Path) -> Dict[str, Any]:
    private_root = root / "evaluator-private"
    records = []
    for path in _files(private_root):
        if path.name == "artifact-manifest.json":
            continue
        data = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(private_root).as_posix(),
                "size": len(data),
                "sha256": digest_bytes(data),
            }
        )
    return {"schema_version": "egv-evaluation-private-artifacts-v1", "files": records}


def _public_manifest(root: Path) -> Dict[str, Any]:
    records = []
    for path in _publishable_files(root):
        if path.name == "artifact-manifest.json":
            continue
        data = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(data),
                "sha256": digest_bytes(data),
            }
        )
    return {"schema_version": "egv-evaluation-public-artifacts-v1", "files": records}


def freeze_evaluation(
    output_root: Union[Path, str],
    *,
    corpus: Optional[EvaluationCorpus] = None,
    prompts: Optional[PromptRegistry] = None,
) -> FreezeReport:
    """Materialize separate trainer, public, and evaluator-private views."""

    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ArtifactError("freeze output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    corpus = _write_seed(root, corpus)
    prompts = prompts or PromptRegistry()
    corpus.validate()
    prompts.validate()

    data_manifest = corpus.manifest()
    # The full manifest contains held-out identifiers and private digests.  A
    # public freeze gets only the closed summary; the complete manifest stays
    # under the evaluator-private boundary.
    _write_json(root / "frozen" / "data-manifest.json", corpus.public_summary())
    _write_json(root / "evaluator-private" / "data-manifest.json", data_manifest)
    _write_json(root / "frozen" / "prompt-manifest.json", prompts.manifest())
    protocol = {
        "schema_version": "egv-evaluation-protocol-v2",
        "data_manifest_digest": digest_for(data_manifest),
        "prompt_manifest_digest": prompts.manifest_digest(),
        "split_counts": {split: len(corpus.split(split)) for split in ("train", "dev", "heldout")},
        "family_counts": corpus.public_summary()["family_counts"],
        "diagnostic_enum": [
            "PASS",
            "WRONG_OUTPUT",
            "SYNTAX_OR_IMPORT",
            "RUNTIME_EXCEPTION",
            "TIMEOUT",
            "RESOURCE_LIMIT",
            "AUTHORITY_DENIED",
            "MUTATION_LOCUS_VIOLATION",
            "PROTOCOL_VIOLATION",
            "INTERNAL_ERROR",
        ],
        "sft_schema_version": "egv-sft-row-v1",
        "max_sequence_length": 4096,
        "sequence_packing": False,
        "model_dependencies": False,
        "authority_backend": "docker-enforced-v1",
        "docker_resource_limits": {
            "network": "none",
            "read_only_root": True,
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "user": "65534:65534",
            "pids_limit": 64,
            "memory": "128m",
            "resource_bound_memory": "64m",
            "tmpfs": "/tmp:ro,noexec,nosuid,nodev,size=16m",
            "timeout_seconds": 2.0,
            "stdout_output_limit_bytes": 65536,
            "resource_buckets": ["UNDER_25", "25_TO_50", "50_TO_75", "75_TO_100", "LIMIT_REACHED", "OUTPUT_LIMIT"],
            "seccomp_default": "SCMP_ACT_ERRNO",
        },
    }
    _write_json(root / "frozen" / "evaluation-protocol.json", protocol)

    trainer_manifest = {
        "schema_version": "egv-evaluation-trainer-data-v2",
        "split_counts": {"train": len(corpus.split("train")), "dev": len(corpus.split("dev"))},
        "tasks": [repo.public_manifest_record() for repo in corpus.trainer_repositories()],
    }
    _write_json(root / "trainer" / "data-manifest.json", trainer_manifest)
    for repo in corpus.trainer_repositories():
        for relative, data in repo.source_files:
            _write_bytes(root / "trainer" / "tasks" / repo.template_id / relative, data)

    public_manifest = {
        "schema_version": "egv-evaluation-public-manifest-v2",
        "protocol_digest": digest_for(protocol),
        "data_manifest_digest": digest_for(data_manifest),
        "prompt_manifest_digest": prompts.manifest_digest(),
        "task_count": 36,
        "split_counts": {"train": 20, "dev": 8, "heldout": 8},
        "family_counts": corpus.public_summary()["family_counts"],
        "hidden_task_ids": "withheld from trainer and public artifacts",
    }
    _write_json(root / "public" / "evaluation-manifest.json", public_manifest)

    _write_json(root / "evaluator-private" / "hidden-manifest.json", corpus.hidden_manifest())
    for repo in corpus.hidden_repositories():
        private_task = root / "evaluator-private" / "heldout" / repo.template_id
        for relative, data in repo.source_files:
            _write_bytes(private_task / relative, data)
        _write_json(
            private_task / "hidden.json",
            repo.private_hidden_record(seed_digest=corpus.secret_seed_digest),
        )
        _write_bytes(private_task / "candidate" / "src" / "task.py", repo.corrected_source)
        _write_json(
            private_task / "candidate" / "candidate-manifest.json",
            {"template_id": repo.template_id, "source_digest": digest_bytes(repo.corrected_source)},
        )

    _write_json(root / "evaluator-private" / "artifact-manifest.json", _private_manifest(root))
    _write_json(root / "artifact-manifest.json", _public_manifest(root))
    public_scan = scan_public_artifacts(root, corpus)

    all_files = _files(root)
    public_files = _publishable_files(root)
    private_files = _files(root / "evaluator-private")
    return FreezeReport(
        data_manifest_digest=digest_for(data_manifest),
        protocol_digest=digest_for(protocol),
        public_scan=public_scan,
        file_count=len(all_files),
        public_file_count=len(public_files),
        private_file_count=len(private_files),
        split_counts={"train": 20, "dev": 8, "heldout": 8},
    )


def assert_byte_identical_regeneration(first_root: Union[Path, str], second_root: Union[Path, str]) -> None:
    if not _same_tree(Path(first_root), Path(second_root)):
        raise ArtifactError("deterministic regeneration produced different artifact bytes")


__all__ = [
    "ArtifactRef",
    "ContentAddressedArtifactStore",
    "FreezeReport",
    "assert_byte_identical_regeneration",
    "freeze_evaluation",
    "scan_public_artifacts",
]
