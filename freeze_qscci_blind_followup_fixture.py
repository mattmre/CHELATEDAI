"""Mechanically split the reviewed blind fixture into capability-isolated frozen inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from validate_qscci_blind_followup_fixture import validate_document


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/research/qscci-blind-followup-fixture-review-draft-v1.json"
SOURCE_SHA256 = "877e2a27ec42afeb2a0b3100f76576aa45fee1c435d08e4748c71efb6338820c"
PARTITIONS = {"DEVELOPMENT": 12, "SELECT": 24, "REPORT": 24}
OUTPUTS = {
    partition: ROOT / f"docs/research/qscci-blind-followup-fixture-v1-{partition.lower()}.json"
    for partition in PARTITIONS
}
LABEL_MAPPING = {
    "NO": {"numeric_label": -1, "target_string": " negative", "token_id": 7968},
    "YES": {"numeric_label": 1, "target_string": " positive", "token_id": 6572},
}
ROOT_KEYS = {
    "fixture_id",
    "label_mapping",
    "near_duplicate_policy",
    "partition",
    "partition_order_index",
    "row_count",
    "rows",
    "schema_version",
    "source_fixture_sha256",
    "status",
    "structural_policy",
    "task",
}


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_partition(source: dict[str, object], partition: str) -> dict[str, object]:
    rows = [row for row in source["rows"] if row["partition"] == partition]
    if len(rows) != PARTITIONS[partition]:
        raise ValueError(f"{partition}: expected {PARTITIONS[partition]} rows, got {len(rows)}")
    document = {
        "fixture_id": f"QSCCI-BF-v1-{partition}",
        "label_mapping": LABEL_MAPPING,
        "near_duplicate_policy": source["near_duplicate_policy"],
        "partition": partition,
        "partition_order_index": source["partition_order"].index(partition),
        "row_count": len(rows),
        "rows": rows,
        "schema_version": 1,
        "source_fixture_sha256": SOURCE_SHA256,
        "status": "FROZEN",
        "structural_policy": source["structural_policy"],
        "task": source["task"],
    }
    if set(document) != ROOT_KEYS:
        raise AssertionError("frozen partition root schema drift")
    return document


def freeze() -> dict[str, str]:
    source_bytes = SOURCE.read_bytes()
    if sha256_bytes(source_bytes) != SOURCE_SHA256:
        raise ValueError("blind authoring source digest mismatch")
    source = json.loads(source_bytes)
    errors = validate_document(source)
    if errors:
        raise ValueError("blind authoring source validation failed: " + "; ".join(errors))

    digests: dict[str, str] = {}
    for partition, output in OUTPUTS.items():
        payload = canonical_bytes(build_partition(source, partition))
        if output.exists():
            if output.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace drifted frozen fixture: {output}")
        else:
            output.write_bytes(payload)
        digests[partition] = sha256_bytes(payload)
    return digests


def main() -> int:
    print(json.dumps(freeze(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
