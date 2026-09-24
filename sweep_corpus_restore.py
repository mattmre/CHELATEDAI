"""Snapshot and restore Qdrant corpus vectors between sweep configurations.

Sedimentation writes adapted vectors back into the shared collection. The
next configuration must see the original corpus, not the previous config's
write. One client stays open so the sweep does not take a second file lock.
"""

from __future__ import annotations

from typing import Any


def snapshot_collection(client: Any, collection_name: str, page_size: int = 256) -> list[dict[str, Any]]:
    """Read every point once. Vectors and payloads are copied as returned."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    records: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        for point in points:
            records.append(
                {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": dict(point.payload or {}),
                }
            )
        if offset is None or not points:
            break
    return records


def restore_collection(
    client: Any,
    collection_name: str,
    records: list[dict[str, Any]],
    page_size: int = 256,
) -> int:
    """Upsert the snapshotted points. An empty snapshot writes nothing."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    if not records:
        return 0
    from qdrant_client.http.models import PointStruct

    written = 0
    for start in range(0, len(records), page_size):
        chunk = records[start:start + page_size]
        points = [
            PointStruct(id=record["id"], vector=record["vector"], payload=record["payload"])
            for record in chunk
        ]
        client.upsert(collection_name=collection_name, points=points)
        written += len(points)
    return written
