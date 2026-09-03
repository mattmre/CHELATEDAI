"""Rebuildable retrieval projections derived from the authoritative ledger."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
import uuid

from .canonical import content_id, digest_for, failure_family_root, validate_sha256
from .errors import OptionalDependencyError, ProjectionError
from .ledger import EvidenceLedger, STALE_DEPENDENT, RETRACTED


EmbeddingFunction = Callable[[Mapping[str, Any]], Sequence[float]]


def qdrant_available() -> bool:
    """Return whether the explicitly optional Qdrant client can be imported."""

    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        return False
    return True


def _require_qdrant() -> Tuple[Any, Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OptionalDependencyError(
            "Qdrant projection was explicitly selected, but 'qdrant-client' is unavailable; "
            "install the optional Qdrant dependency or select InMemoryProjection explicitly"
        ) from exc
    return QdrantClient, models, qdrant_available


def _qdrant_client_for_location(QdrantClient: Any, location: Union[str, Path]) -> Any:
    """Construct the correct Qdrant client for a local path or remote URL.

    qdrant-client accepts both ``location`` and ``path`` in its public
    constructor, but a filesystem path supplied as ``location`` is treated as
    a remote endpoint.  Keep the Slice 2 location contract explicit so a
    local projection never silently turns into a network client.
    """

    if isinstance(location, Path):
        return QdrantClient(path=str(location))
    if not isinstance(location, str) or not location:
        raise ProjectionError("Qdrant location must be a non-empty path, :memory:, or HTTP(S) URL")
    if location == ":memory:":
        return QdrantClient(location=location)

    parsed = urlparse(location)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https"}:
        if not parsed.netloc:
            raise ProjectionError(f"Qdrant {scheme.upper()} location must include a host")
        return QdrantClient(url=location)
    if scheme or "://" in location:
        raise ProjectionError(
            "unsupported Qdrant location; use :memory:, a filesystem path, or an explicit HTTP(S) URL"
        )
    return QdrantClient(path=location)


def isolated_collection_name(campaign_id: str, arm: str, run_id: str) -> str:
    """Derive an isolated Qdrant namespace without exposing raw IDs in paths."""

    return content_id("egv", {"campaign_id": campaign_id, "arm": arm, "run_id": run_id})


def qdrant_point_uuid(canonical_point_id: str) -> str:
    """Map a canonical content ID to a deterministic Qdrant-compatible UUID.

    The evidence core keeps its full SHA-256 ``point_<digest>`` ID as the
    canonical identity.  Qdrant accepts only integers or UUIDs for point IDs,
    so use the first 128 bits of a SHA-256 digest of that canonical ID and set
    the RFC 4122 variant/version bits.  This retains 122 collision-resistant
    bits while producing a portable UUID string accepted by Qdrant's local and
    server clients.
    """

    if not isinstance(canonical_point_id, str) or not canonical_point_id:
        raise ProjectionError("canonical point ID must be a non-empty string")
    digest = bytes.fromhex(digest_for(canonical_point_id))
    raw = bytearray(digest[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


@dataclass(frozen=True)
class ProjectionPoint:
    point_id: str
    vector: Tuple[float, ...]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"point_id": self.point_id, "vector": list(self.vector), "payload": dict(self.payload)}


def _validate_vector(vector: Sequence[float]) -> Tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise ProjectionError("embedding must be a finite numeric sequence")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise ProjectionError("embedding must be a finite numeric sequence") from exc
    if not values:
        raise ProjectionError("embedding must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ProjectionError("embedding contains a non-finite value")
    return values


class InMemoryProjection:
    """Explicit dependency-free projection backend for tests and local replay."""

    backend_name = "memory"

    def __init__(self, *, collection_name: str = "egv-memory") -> None:
        self.collection_name = collection_name
        self.points: Dict[str, ProjectionPoint] = {}
        self.projection_generation: Optional[str] = None
        self.ledger_head_hash: Optional[str] = None
        self.rebuild_count = 0

    def rebuild(
        self,
        ledger: EvidenceLedger,
        embedding_fn: EmbeddingFunction,
        *,
        embedding_model_revision: str,
        campaign_id: Optional[str] = None,
        arm: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build all points from one committed ledger snapshot."""

        if not callable(embedding_fn):
            raise ProjectionError("embedding_fn must be callable")
        source_events = ledger.current_valid_events()
        if campaign_id is not None:
            source_events = [event for event in source_events if event.get("campaign_id") == campaign_id]
        if arm is not None or run_id is not None:
            runs_by_id = {row["run_id"]: row["arm"] for row in ledger.connection.execute("SELECT run_id,arm FROM runs")}
            source_events = [
                event
                for event in source_events
                if (arm is None or runs_by_id.get(event.get("run_id")) == arm)
                and (run_id is None or event.get("run_id") == run_id)
            ]
        prepared: List[ProjectionPoint] = []
        for event in source_events:
            event_disposition = ledger.event_disposition(event["event_id"])
            if event_disposition in {RETRACTED, STALE_DEPENDENT}:
                continue
            raw_payload = event.get("payload")
            if raw_payload is None:
                # A large private blob cannot be embedded without its blob store;
                # omitting it would hide data, so fail rather than silently project
                # a partial view.
                raise ProjectionError(f"event {event['event_id']} payload is unavailable for projection")
            payload = raw_payload
            # EvidenceLedger.ingest_receipt stores the authoritative receipt
            # event as {receipt: <signed receipt>, receipt_hash: <digest>}.
            # Projection must validate an INTERNAL_ERROR receipt's signed
            # failure tuple, not the transport wrapper (which has no
            # diagnostic fields and could otherwise make a dummy root look
            # acceptable). Ordinary signed receipts repeat task-family,
            # locus, and rule context for correlation; that common context is
            # not a failure-root claim.
            if event.get("event_type") == "RECEIPT":
                if not isinstance(raw_payload, Mapping) or not isinstance(raw_payload.get("receipt"), Mapping):
                    raise ProjectionError("RECEIPT events must contain a nested signed receipt")
                payload = dict(raw_payload["receipt"])
            if not isinstance(payload, Mapping):
                raise ProjectionError(f"event {event['event_id']} payload is not an object")
            vector = _validate_vector(embedding_fn(payload))
            explicit_failure_root = payload.get("failure_family_root")
            infrastructure_incident_id = payload.get("infrastructure_incident_id")
            failure_root_key_present = "failure_family_root" in payload
            incident_key_present = "infrastructure_incident_id" in payload
            root_fields = ("task_family", "diagnostic_enum", "normalized_public_locus", "public_rule_id")
            diagnostic = payload.get("diagnostic_enum")
            # Signed receipts carry task-family/locus/rule context on every
            # receipt.  Those common fields are not a failure-root claim:
            # receipts reserve roots and incidents exclusively for
            # INTERNAL_ERROR.  Treat any explicit non-internal claim as a
            # malformed event instead of silently dropping it.
            if diagnostic != "INTERNAL_ERROR":
                if failure_root_key_present or incident_key_present:
                    raise ProjectionError("failure roots and incidents are reserved for INTERNAL_ERROR")
                root_is_claimed = False
            else:
                root_is_claimed = True
            canonical_failure_root = None
            if root_is_claimed:
                missing = [
                    key
                    for key in root_fields + ("infrastructure_incident_id", "failure_family_root")
                    if key not in payload
                ]
                if missing:
                    raise ProjectionError(
                        "failure-family projection fields are incomplete: {}".format(",".join(missing))
                    )
                if explicit_failure_root is None:
                    raise ProjectionError("diagnostic events must carry an explicit failure_family_root")
                try:
                    canonical_failure_root = failure_family_root(
                        payload["task_family"],
                        payload["diagnostic_enum"],
                        payload["normalized_public_locus"],
                        payload["public_rule_id"],
                        infrastructure_incident_id=infrastructure_incident_id,
                    )
                except Exception as exc:
                    raise ProjectionError("failure-family projection fields are not canonical") from exc
                if explicit_failure_root is not None:
                    try:
                        explicit_failure_root = validate_sha256(explicit_failure_root, "failure_family_root")
                    except Exception as exc:
                        raise ProjectionError("failure_family_root must be a full SHA-256 digest") from exc
                    if explicit_failure_root != canonical_failure_root:
                        raise ProjectionError("failure_family_root does not match the canonical public tuple")
            point_id = content_id(
                "point",
                {"event_id": event["event_id"], "payload_hash": event["payload_hash"], "embedding_model_revision": embedding_model_revision},
            )
            point_payload = {
                "canonical_point_id": point_id,
                "source_event_id": event["event_id"],
                "source_payload_hash": event["payload_hash"],
                "validity_state": event_disposition,
                "failure_family_root": canonical_failure_root,
                "projection_generation": None,
                "embedding_model_revision": embedding_model_revision,
            }
            prepared.append(ProjectionPoint(point_id, vector, point_payload))
        generation = digest_for(
            {
                "ledger_head_hash": ledger.ledger_head_hash(),
                "embedding_model_revision": embedding_model_revision,
                "points": [point.to_dict() for point in sorted(prepared, key=lambda point: point.point_id)],
            }
        )
        points = {
            point.point_id: ProjectionPoint(
                point.point_id,
                point.vector,
                {**point.payload, "projection_generation": generation},
            )
            for point in prepared
        }
        # Only replace the live projection after every embedding and generation
        # calculation succeeds. The ledger was already committed before this call.
        self.points = points
        self.projection_generation = generation
        self.ledger_head_hash = ledger.ledger_head_hash()
        self.rebuild_count += 1
        return self.manifest()

    def manifest(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "collection_name": self.collection_name,
            "point_count": len(self.points),
            "projection_generation": self.projection_generation,
            "ledger_head_hash": self.ledger_head_hash,
            "point_ids": sorted(self.points),
        }

    def delete(self) -> None:
        self.points = {}
        self.projection_generation = None
        self.ledger_head_hash = None

    def get(self, point_id: str) -> Optional[ProjectionPoint]:
        return self.points.get(point_id)

    def query(self, vector: Sequence[float], *, limit: int = 10) -> List[Tuple[ProjectionPoint, float]]:
        query = _validate_vector(vector)
        if not self.points:
            return []
        if any(len(point.vector) != len(query) for point in self.points.values()):
            raise ProjectionError("query vector dimension does not match projection")
        query_norm = math.sqrt(sum(value * value for value in query))
        if query_norm == 0:
            raise ProjectionError("query vector must have non-zero norm")
        scored = []
        for point in self.points.values():
            point_norm = math.sqrt(sum(value * value for value in point.vector))
            score = 0.0 if point_norm == 0 else sum(a * b for a, b in zip(query, point.vector)) / (query_norm * point_norm)
            scored.append((point, score))
        scored.sort(key=lambda item: (-item[1], item[0].point_id))
        return scored[: max(0, limit)]

    def validate_hit(self, ledger: EvidenceLedger, point: ProjectionPoint) -> bool:
        if self.projection_generation is None or point.payload.get("projection_generation") != self.projection_generation:
            return False
        event_id = point.payload.get("source_event_id")
        row = ledger.connection.execute("SELECT payload_hash FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None or row[0] != point.payload.get("source_payload_hash"):
            return False
        return ledger.event_disposition(event_id) not in {RETRACTED, STALE_DEPENDENT}


class QdrantProjection:
    """Qdrant-backed disposable projection with explicit optional dependency."""

    backend_name = "qdrant"

    def __init__(
        self,
        *,
        location: Union[str, Path],
        collection_name: str,
        client: Any = None,
    ) -> None:
        QdrantClient, _models, _available = _require_qdrant()
        self.collection_name = collection_name
        self.location = str(location)
        self.client = client if client is not None else _qdrant_client_for_location(QdrantClient, location)
        self.projection_generation: Optional[str] = None
        self.ledger_head_hash: Optional[str] = None
        self._models = None

    def _models_import(self) -> Any:
        if self._models is None:
            try:
                from qdrant_client import models
            except ImportError as exc:  # pragma: no cover
                raise OptionalDependencyError("Qdrant models are unavailable") from exc
            self._models = models
        return self._models

    def _ensure_collection(self, vector_size: int) -> None:
        models = self._models_import()
        exists = self.client.collection_exists(self.collection_name)
        if exists:
            info = self.client.get_collection(self.collection_name)
            current_size = getattr(getattr(info, "config", None), "params", None)
            current_size = getattr(current_size, "vectors", None)
            current_size = getattr(current_size, "size", None)
            if current_size is not None and int(current_size) != vector_size:
                raise ProjectionError("existing Qdrant collection has a different vector dimension")
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def rebuild(
        self,
        ledger: EvidenceLedger,
        embedding_fn: EmbeddingFunction,
        *,
        embedding_model_revision: str,
        campaign_id: Optional[str] = None,
        arm: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Compute through the dependency-free implementation first. It gives us
        # the exact point IDs/generation that Qdrant must receive.
        memory = InMemoryProjection(collection_name=self.collection_name)
        manifest = memory.rebuild(
            ledger,
            embedding_fn,
            embedding_model_revision=embedding_model_revision,
            campaign_id=campaign_id,
            arm=arm,
            run_id=run_id,
        )
        points = list(memory.points.values())
        try:
            dimension = len(points[0].vector) if points else None
            # Rebuild means exact replacement. Leaving old Qdrant points in
            # place would make a projection silently depend on prior state and
            # violate ledger-only reproducibility.
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            if dimension is not None:
                self._ensure_collection(dimension)
                models = self._models_import()
                structs = [
                    models.PointStruct(
                        id=qdrant_point_uuid(point.point_id),
                        vector=list(point.vector),
                        payload=point.payload,
                    )
                    for point in points
                ]
                if structs:
                    self.client.upsert(collection_name=self.collection_name, points=structs, wait=True)
            self.projection_generation = manifest["projection_generation"]
            self.ledger_head_hash = manifest["ledger_head_hash"]
        except Exception as exc:
            ledger.enqueue_projection_rebuild(
                projection_generation=manifest.get("projection_generation"),
                reason_code=type(exc).__name__,
            )
            raise ProjectionError(f"Qdrant projection write failed; ledger remains authoritative: {exc}") from exc
        return {**manifest, "backend": self.backend_name}

    def delete(self) -> None:
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
        except Exception as exc:
            raise ProjectionError(f"Qdrant collection deletion failed: {exc}") from exc
        self.projection_generation = None
        self.ledger_head_hash = None

    def validate_hit(self, ledger: EvidenceLedger, hit: Any) -> bool:
        payload = getattr(hit, "payload", None) or {}
        if payload.get("projection_generation") != self.projection_generation:
            return False
        canonical_point_id = payload.get("canonical_point_id")
        if not isinstance(canonical_point_id, str):
            return False
        try:
            if str(getattr(hit, "id", "")) != qdrant_point_uuid(canonical_point_id):
                return False
        except ProjectionError:
            return False
        event_id = payload.get("source_event_id")
        row = ledger.connection.execute("SELECT payload_hash FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None or row[0] != payload.get("source_payload_hash"):
            return False
        return ledger.event_disposition(event_id) not in {RETRACTED, STALE_DEPENDENT}


def create_projection(
    *,
    backend: str,
    collection_name: str,
    location: Optional[Union[str, Path]] = None,
) -> Union[InMemoryProjection, QdrantProjection]:
    """Create a named backend; never silently replace Qdrant with memory."""

    if backend == "memory":
        return InMemoryProjection(collection_name=collection_name)
    if backend == "qdrant":
        if location is None:
            raise ProjectionError("Qdrant backend requires an explicit location")
        return QdrantProjection(location=location, collection_name=collection_name)
    raise ProjectionError(f"unknown projection backend: {backend!r}")


__all__ = [
    "EmbeddingFunction",
    "InMemoryProjection",
    "ProjectionPoint",
    "QdrantProjection",
    "create_projection",
    "isolated_collection_name",
    "qdrant_point_uuid",
    "qdrant_available",
]
