"""Build a path-free identity receipt for a pinned Hugging Face snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


SCHEMA_VERSION = "egv-hf-snapshot-identity-v1"
_LFS_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIRECT_HASH_CEILING = 128 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(path: Path, size: int) -> tuple[str, str]:
    if path.is_symlink():
        blob_id = Path(os.readlink(path)).name
        if _LFS_SHA256.fullmatch(blob_id) is not None:
            return blob_id, "hugging-face-lfs-sha256"
    if size > _DIRECT_HASH_CEILING:
        raise RuntimeError("large snapshot file lacks a sha256 LFS object identity")
    return _sha256_file(path), "direct-sha256"


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical(value))
        handle.flush()
    temporary.replace(path)


def build_identity(snapshot: Path, repository: str, revision: str) -> dict[str, Any]:
    if not snapshot.is_dir():
        raise RuntimeError("snapshot directory is unavailable")
    entries = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot).as_posix()
        size = path.stat().st_size
        content_sha256, source = _content_digest(path, size)
        entries.append(
            {
                "path": relative,
                "size_bytes": size,
                "content_sha256": content_sha256,
                "identity_source": source,
            }
        )
    if not entries:
        raise RuntimeError("snapshot contains no files")
    weight_entries = [entry for entry in entries if re.fullmatch(r"model-\d+-of-\d+\.safetensors", entry["path"])]
    if not weight_entries:
        raise RuntimeError("snapshot contains no sharded model weights")
    selected_names = {
        "LICENSE",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    selected = {
        entry["path"]: entry["content_sha256"]
        for entry in entries
        if entry["path"] in selected_names
    }
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "revision": revision,
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "content_manifest_digest": _sha256_bytes(_canonical(entries)),
        "weight_shard_count": len(weight_entries),
        "weight_bytes": sum(entry["size_bytes"] for entry in weight_entries),
        "weight_manifest_digest": _sha256_bytes(_canonical(weight_entries)),
        "selected_file_sha256": selected,
        "identity_algorithm": (
            "sorted path,size,sha256 entries; immutable Hugging Face LFS object IDs "
            "for large files; direct SHA256 for other files"
        ),
    }
    core["receipt_digest"] = _sha256_bytes(_canonical(core))
    return core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_identity(args.snapshot, args.repository, args.revision)
    if args.output is None:
        print(_canonical(report).decode("utf-8"), end="")
    else:
        _atomic_write(args.output, report)
        print(json.dumps({"receipt_digest": report["receipt_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
