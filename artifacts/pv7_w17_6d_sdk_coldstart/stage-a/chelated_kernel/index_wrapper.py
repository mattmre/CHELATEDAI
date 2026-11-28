"""Tenant-partitioned ZeroMu + CSLS index wrap with a fail-closed TRA-1 check.

Resident vectors stay unmodified. Each document carries an 8-byte sidecar:
little-endian float32 centered norm, then little-endian float32 CSLS radius.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from chelated_kernel.errors import TenantIsolationError, Tra1VerificationError
from chelated_kernel.profiles import IndexProfile, get_index_profile, require_tenant_id

LIST_VECTORS_SQL = (
    "SELECT id, embedding, tenant_id "
    "FROM chelated_vectors "
    "WHERE tenant_id = %s "
    "ORDER BY id ASC"
)

_INT64_MIN = -9223372036854775808
_INT64_MAX = 9223372036854775807


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    tenant_id: str


@dataclass(frozen=True)
class Tra1Contract:
    tenant_id: str
    old_dim: int
    new_dim: int
    rectangular: bool
    fit_count: int
    validation_count: int
    confidence: float
    confidence_lcb: float
    sigma_min: float
    ortho_error: float
    sidecar_bytes_per_doc: int
    doc_count: int
    digest_hex: str
    profile: str
    profile_version: str
    bootstrap_resamples: int
    llm_write_tokens: int


def _as_matrix(value: object, *, width: int, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise Tra1VerificationError("anchor_dim", f"{what} must have shape (n, {width})")
    if not np.isfinite(arr).all():
        raise Tra1VerificationError("non_finite", f"{what} contains NaN or Inf")
    return arr


def _as_vector(value: object, *, width: int, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (width,):
        raise Tra1VerificationError("vector_dim", f"{what} must have length {width}")
    if not np.isfinite(arr).all():
        raise Tra1VerificationError("non_finite", f"{what} contains NaN or Inf")
    return arr


def _bootstrap_lcb(values: np.ndarray, seed: int, *, resamples: int, alpha: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise Tra1VerificationError("bootstrap", "validation cosine sample is empty")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF]))
    drawn = rng.integers(0, arr.shape[0], size=(int(resamples), arr.shape[0]))
    means = arr[drawn].mean(axis=1)
    return float(np.quantile(means, alpha))


def _fit_map(old_fit: np.ndarray, new_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu_old = old_fit.mean(axis=0)
    mu_new = new_fit.mean(axis=0)
    old_c = old_fit - mu_old
    new_c = new_fit - mu_new
    if float(np.linalg.norm(old_c)) <= 1e-8 or float(np.linalg.norm(new_c)) <= 1e-8:
        raise Tra1VerificationError("degenerate_anchors", "centered anchor cloud has no spread")
    cross = new_c.T @ old_c
    singular = np.linalg.svd(cross, compute_uv=False)
    sigma_min = float(singular[-1]) if singular.size else 0.0
    if sigma_min <= 1e-8:
        raise Tra1VerificationError("degenerate_spectrum", f"sigma_min={sigma_min}")
    u_mat, _s, vt_mat = np.linalg.svd(cross, full_matrices=False)
    mapping = u_mat @ vt_mat
    return mapping, mu_old, mu_new, sigma_min


def _ortho_error(mapping: np.ndarray, old_dim: int, new_dim: int) -> float:
    if new_dim >= old_dim:
        gram = mapping.T @ mapping
        eye = np.eye(old_dim)
    else:
        gram = mapping @ mapping.T
        eye = np.eye(new_dim)
    return float(np.max(np.abs(gram - eye)))


def _row_cosines(mapped: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(mapped * target, axis=1)
    denom = np.linalg.norm(mapped, axis=1) * np.linalg.norm(target, axis=1)
    return numerator / np.maximum(denom, 1e-12)


def _csls_radius(
    reference: np.ndarray,
    docs: np.ndarray,
    mu: np.ndarray,
    den: np.ndarray,
    csls_k: int,
) -> np.ndarray:
    ref_norm = np.linalg.norm(reference, axis=1)
    if np.any(ref_norm <= 1e-12):
        raise Tra1VerificationError("degenerate_reference", "a CSLS reference query has zero norm")
    mu_dots = reference @ mu
    dots = reference @ docs.T
    cos = (dots - mu_dots[:, None]) / (ref_norm[:, None] * den[None, :])
    k_eff = max(1, min(int(csls_k), cos.shape[0]))
    if k_eff < cos.shape[0]:
        top = np.partition(cos, cos.shape[0] - k_eff, axis=0)[-k_eff:, :]
    else:
        top = cos
    return top.mean(axis=0)


def _pack_sidecar(den: np.ndarray, radius: np.ndarray) -> bytes:
    if den.shape != radius.shape:
        raise Tra1VerificationError("sidecar_shape", "norm and radius length differ")
    out = bytearray()
    for norm_i, radius_i in zip(den.tolist(), radius.tolist()):
        out.extend(struct.pack("<ff", float(norm_i), float(radius_i)))
    return bytes(out)


def _digest(
    *,
    tenant_id: str,
    old_dim: int,
    new_dim: int,
    fit_count: int,
    validation_count: int,
    mapping: np.ndarray,
    mu_old: np.ndarray,
    mu_new: np.ndarray,
    confidence: float,
    doc_ids: list[str],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(tenant_id.encode("utf-8"))
    hasher.update(struct.pack("<iiii", old_dim, new_dim, fit_count, validation_count))
    hasher.update(np.rint(mapping * 1_000_000_000.0).astype(np.int64).tobytes())
    hasher.update(np.rint(mu_old * 1_000_000_000.0).astype(np.int64).tobytes())
    hasher.update(np.rint(mu_new * 1_000_000_000.0).astype(np.int64).tobytes())
    scaled = int(round(confidence * 1_000_000_000.0))
    if scaled < _INT64_MIN or scaled > _INT64_MAX:
        raise Tra1VerificationError("digest_range", "confidence scale exceeds int64")
    hasher.update(struct.pack("<q", scaled))
    hasher.update("\n".join(doc_ids).encode("utf-8"))
    return hasher.hexdigest()


def _tenant_filter(tenant_id: str) -> dict[str, list[dict[str, object]]]:
    return {"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]}


def _record_parts(key: object, value: object, default_tenant: str) -> tuple[str, object | None, str]:
    if isinstance(value, dict):
        tenant = value.get("tenant_id", default_tenant)
        vector = value.get("vector", value.get("embedding"))
        doc_id = str(value.get("id", key))
        return doc_id, vector, str(tenant)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return str(key), value[0], value[1]
    return str(key), value, default_tenant


def _rows_from_mapping(backend: dict[object, object], tenant_id: str) -> tuple[list[tuple[str, np.ndarray]], int]:
    rows: list[tuple[str, np.ndarray]] = []
    rejected = 0
    for key, value in backend.items():
        doc_id, vector, tenant = _record_parts(key, value, tenant_id)
        if tenant != tenant_id:
            rejected += 1
            continue
        rows.append((doc_id, _as_vector(vector, width=_peek_width(vector), what=doc_id)))
    return rows, rejected


def _peek_width(vector: object) -> int:
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    return int(arr.shape[0])


def _rows_from_callable(backend: object, tenant_id: str, old_dim: int) -> tuple[list[tuple[str, np.ndarray]], int]:
    listed = backend({"op": "list", "tenant_id": tenant_id})  # type: ignore[operator]
    if not isinstance(listed, list):
        raise Tra1VerificationError("backend_list", "callable backend list op must return a list")
    rows: list[tuple[str, np.ndarray]] = []
    rejected = 0
    for item in listed:
        if not isinstance(item, dict):
            raise Tra1VerificationError("backend_row", "callable backend rows must be dicts")
        tenant = str(item.get("tenant_id", tenant_id))
        if tenant != tenant_id:
            rejected += 1
            continue
        doc_id = str(item["id"])
        rows.append((doc_id, _as_vector(item.get("vector", item.get("embedding")), width=old_dim, what=doc_id)))
    return rows, rejected


def _qdrant_point(point: object) -> tuple[str, object, str | None]:
    if isinstance(point, dict):
        payload = point.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        tenant = payload.get("tenant_id")
        return str(point.get("id")), point.get("vector"), None if tenant is None else str(tenant)
    payload = getattr(point, "payload", {}) or {}
    tenant = payload.get("tenant_id") if isinstance(payload, dict) else None
    return str(getattr(point, "id", "")), getattr(point, "vector", None), None if tenant is None else str(tenant)


def _rows_from_qdrant(backend: object, tenant_id: str, old_dim: int) -> tuple[list[tuple[str, np.ndarray]], int]:
    filt = _tenant_filter(tenant_id)
    offset: object = None
    rows: list[tuple[str, np.ndarray]] = []
    rejected = 0
    seen = 0
    while True:
        result = backend.scroll(scroll_filter=filt, limit=256, offset=offset)  # type: ignore[attr-defined]
        if isinstance(result, tuple):
            points = result[0]
            offset = result[1] if len(result) > 1 else None
        else:
            points = result
            offset = None
        if not points:
            break
        for point in points:
            seen += 1
            doc_id, vector, tenant = _qdrant_point(point)
            if tenant != tenant_id:
                rejected += 1
                continue
            rows.append((doc_id, _as_vector(vector, width=old_dim, what=doc_id)))
        if offset is None:
            break
        if seen > 1_000_000:
            raise Tra1VerificationError("backend_page", "qdrant scroll did not terminate")
    return rows, rejected


def _sql_tenant_guard(sql: str) -> None:
    folded = " ".join(sql.lower().split())
    if "where tenant_id = %s" not in folded:
        raise TenantIsolationError("vector SQL must bind tenant_id")
    if " or " in folded or "1=1" in folded.replace(" ", ""):
        raise TenantIsolationError("vector SQL must not widen the tenant predicate")


def _rows_from_sql(backend: object, tenant_id: str, old_dim: int) -> tuple[list[tuple[str, np.ndarray]], int]:
    _sql_tenant_guard(LIST_VECTORS_SQL)
    fetched = backend.execute(LIST_VECTORS_SQL, (tenant_id,))  # type: ignore[attr-defined]
    rows: list[tuple[str, np.ndarray]] = []
    rejected = 0
    for record in fetched:
        doc_id, embedding, tenant = record[0], record[1], record[2]
        if str(tenant) != tenant_id:
            rejected += 1
            continue
        blob = bytes(embedding)
        expect = old_dim * 8
        if len(blob) != expect:
            raise Tra1VerificationError("embedding_blob", f"{doc_id} blob is {len(blob)} bytes, expected {expect}")
        vector = np.frombuffer(blob, dtype="<f8").astype(np.float64, copy=True)
        rows.append((str(doc_id), vector))
    return rows, rejected


def _rows_from_faiss(backend: object, tenant_id: str, old_dim: int) -> tuple[list[tuple[str, np.ndarray]], int]:
    total = int(getattr(backend, "ntotal"))
    ids = list(getattr(backend, "ids"))
    tenants = list(getattr(backend, "tenant_ids"))
    if len(ids) != total or len(tenants) != total:
        raise Tra1VerificationError("faiss_meta", "ids and tenant_ids must align with ntotal")
    probe = np.zeros((1, old_dim), dtype=np.float32)
    found = backend.search(probe, 1)  # type: ignore[attr-defined]
    foreign_hits: set[int] = set()
    indices = found[1] if isinstance(found, tuple) and len(found) >= 2 else []
    for hit in np.asarray(indices, dtype=np.int64).reshape(-1).tolist():
        if 0 <= hit < total and str(tenants[hit]) != tenant_id:
            foreign_hits.add(hit)
    rows: list[tuple[str, np.ndarray]] = []
    rejected = 0
    for index in range(total):
        if str(tenants[index]) != tenant_id or index in foreign_hits:
            rejected += 1
            continue
        vector = backend.reconstruct(index)  # type: ignore[attr-defined]
        rows.append((str(ids[index]), _as_vector(vector, width=old_dim, what=str(ids[index]))))
    return rows, rejected


def _collect_rows(backend: object, tenant_id: str, old_dim: int) -> tuple[list[tuple[str, np.ndarray]], int]:
    if isinstance(backend, dict):
        rows, rejected = _rows_from_mapping(backend, tenant_id)
    elif callable(backend):
        rows, rejected = _rows_from_callable(backend, tenant_id, old_dim)
    elif hasattr(backend, "execute"):
        rows, rejected = _rows_from_sql(backend, tenant_id, old_dim)
    elif hasattr(backend, "scroll") and hasattr(backend, "search"):
        rows, rejected = _rows_from_qdrant(backend, tenant_id, old_dim)
    elif hasattr(backend, "reconstruct") and hasattr(backend, "search") and hasattr(backend, "ntotal"):
        rows, rejected = _rows_from_faiss(backend, tenant_id, old_dim)
    else:
        raise Tra1VerificationError(
            "backend_kind",
            "backend must be a dict, callable, SQL execute(), Qdrant-style scroll, or FAISS-style reconstruct",
        )
    widths = {row[1].shape[0] for row in rows}
    if any(width != old_dim for width in widths):
        raise Tra1VerificationError("resident_dim", f"resident vectors must be old_dim={old_dim}")
    rows.sort(key=lambda item: item[0])
    return rows, rejected


def _parse_anchors(
    unlabeled_anchors: object,
    *,
    old_dim: int,
    new_dim: int,
    tenant_id: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not isinstance(unlabeled_anchors, (list, tuple)):
        raise Tra1VerificationError("anchors", "unlabeled_anchors must be a sequence of pairs")
    old_rows: list[np.ndarray] = []
    new_rows: list[np.ndarray] = []
    ids: list[str] = []
    for index, anchor in enumerate(unlabeled_anchors):
        if isinstance(anchor, dict):
            if "relevance" in anchor or "qrels" in anchor or "label" in anchor:
                raise Tra1VerificationError("labeled_anchor", "unlabeled_anchors must not carry relevance labels")
            if "tenant_id" in anchor and str(anchor["tenant_id"]) != tenant_id:
                raise TenantIsolationError("anchor tenant_id does not match the wrapper tenant")
            old_value = anchor.get("old", anchor.get("old_vector"))
            new_value = anchor.get("new", anchor.get("new_vector"))
            anchor_id = str(anchor.get("id", f"a-{index:04d}"))
        elif isinstance(anchor, tuple) and len(anchor) == 2:
            old_value, new_value = anchor
            anchor_id = f"a-{index:04d}"
        else:
            raise Tra1VerificationError("anchors", "each anchor must be a pair or a dict with old/new vectors")
        old_rows.append(_as_vector(old_value, width=old_dim, what=f"{anchor_id}.old"))
        new_rows.append(_as_vector(new_value, width=new_dim, what=f"{anchor_id}.new"))
        ids.append(anchor_id)
    if not old_rows:
        raise Tra1VerificationError("anchors", "at least one anchor is required")
    return np.stack(old_rows), np.stack(new_rows), ids


class WrappedIndex:
    """Exact ZeroMu + CSLS search over one tenant partition."""

    def __init__(
        self,
        *,
        tenant_id: str,
        profile: IndexProfile,
        contract: Tra1Contract,
        doc_ids: list[str],
        docs: np.ndarray,
        mu_old: np.ndarray,
        mu_new: np.ndarray,
        mapping: np.ndarray,
        den: np.ndarray,
        radius: np.ndarray,
        sidecar: bytes,
        anchor_old: np.ndarray,
        anchor_new: np.ndarray,
        rejected_foreign_records: int,
        old_dim: int,
        new_dim: int,
        csls_k: int,
    ) -> None:
        self.tenant_id = tenant_id
        self.profile = profile
        self.contract = contract
        self._doc_ids = doc_ids
        self._id_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        self._docs = docs
        self._mu_old = mu_old
        self._mu_new = mu_new
        self._mapping = mapping
        self._den = den
        self._radius = radius
        self._sidecar = sidecar
        self._anchor_old = anchor_old
        self._anchor_new = anchor_new
        self.rejected_foreign_records = int(rejected_foreign_records)
        self.unauthorized_reads = 0
        self.llm_write_tokens = 0
        self._old_dim = old_dim
        self._new_dim = new_dim
        self._csls_k = csls_k

    def __len__(self) -> int:
        return len(self._doc_ids)

    def sidecar_bytes(self, doc_id: str | None = None) -> bytes:
        if doc_id is None:
            return self._sidecar
        try:
            index = self._id_index[str(doc_id)]
        except KeyError as exc:
            raise TenantIsolationError("doc is not in this tenant partition") from exc
        start = index * 8
        return self._sidecar[start : start + 8]

    def search(self, query: object, k: int = 10, **rejected: object) -> list[Hit]:
        if rejected:
            raise TenantIsolationError("search does not accept a tenant override")
        k_eff = int(k)
        if k_eff < 1:
            raise Tra1VerificationError("k", "k must be >= 1")
        if not self._doc_ids:
            return []
        query_vec = _as_vector(query, width=self._new_dim, what="query")
        resident = (query_vec - self._mu_new) @ self._mapping
        q_norm = float(np.linalg.norm(resident))
        if q_norm <= 1e-12:
            raise Tra1VerificationError("query_degenerate", "mapped query has zero norm")
        mu_dot = float(resident @ self._mu_old)
        dots = self._docs @ resident
        cosine = (dots - mu_dot) / (q_norm * self._den)
        k_q = max(1, min(self._csls_k, cosine.shape[0]))
        if k_q < cosine.shape[0]:
            top = np.partition(cosine, cosine.shape[0] - k_q)[-k_q:]
        else:
            top = cosine
        r_q = float(np.mean(top))
        scores = 2.0 * cosine - r_q - self._radius
        order = sorted(range(scores.shape[0]), key=lambda index: (-float(scores[index]), self._doc_ids[index]))
        hits: list[Hit] = []
        for index in order[:k_eff]:
            hits.append(Hit(self._doc_ids[index], float(scores[index]), self.tenant_id))
        return hits


def wrap_index(
    backend: object,
    *,
    old_dim: int,
    new_dim: int,
    unlabeled_anchors: object,
    tenant_id: str = "default",
    profile: str = "enterprise_search",
) -> WrappedIndex:
    """Fit ZeroMu, verify TRA-1, and bind an 8-byte CSLS sidecar to one tenant."""

    tenant = require_tenant_id(tenant_id)
    spec = get_index_profile(profile)
    if isinstance(old_dim, bool) or isinstance(new_dim, bool):
        raise Tra1VerificationError("dim", "old_dim and new_dim must be positive integers")
    if not isinstance(old_dim, int) or not isinstance(new_dim, int) or old_dim < 1 or new_dim < 1:
        raise Tra1VerificationError("dim", "old_dim and new_dim must be positive integers")
    old_anchors, new_anchors, _anchor_ids = _parse_anchors(
        unlabeled_anchors,
        old_dim=old_dim,
        new_dim=new_dim,
        tenant_id=tenant,
    )
    need = spec.min_fit_count + spec.min_validation_count
    if old_anchors.shape[0] < need:
        raise Tra1VerificationError(
            "fit_count",
            f"need {need} unlabeled anchors ({spec.min_fit_count} fit + {spec.min_validation_count} validation)",
        )
    n_total = old_anchors.shape[0]
    min_full_rank = min(old_dim, new_dim) + 1
    preferred_val = max(spec.min_validation_count, n_total // 5)
    max_val = n_total - max(spec.min_fit_count, min_full_rank)
    if max_val >= spec.min_validation_count:
        val_count = min(preferred_val, max_val)
    else:
        val_count = spec.min_validation_count
    fit_count = n_total - val_count
    fit_old = old_anchors[:fit_count]
    fit_new = new_anchors[:fit_count]
    val_old = old_anchors[fit_count:]
    val_new = new_anchors[fit_count:]
    mapping, mu_old, mu_new, sigma_min = _fit_map(fit_old, fit_new)
    ortho_error = _ortho_error(mapping, old_dim, new_dim)
    if ortho_error > 1e-5:
        raise Tra1VerificationError("ortho", f"Stiefel residual {ortho_error}")
    # Score the forward pairing. A new_dim < old_dim map is a projection in the
    # resident space, so cosine against the old anchors cannot reach 1.
    predicted = (val_old - mu_old) @ mapping.T
    cosines = _row_cosines(predicted, val_new - mu_new)
    confidence = float(np.mean(cosines))
    seed = int.from_bytes(hashlib.sha256(tenant.encode("utf-8")).digest()[:4], "little")
    confidence_lcb = _bootstrap_lcb(
        cosines,
        seed,
        resamples=spec.bootstrap_resamples,
        alpha=spec.bootstrap_alpha,
    )
    floor = spec.minimum_route_confidence
    if confidence < floor or confidence_lcb < floor:
        raise Tra1VerificationError(
            "confidence",
            f"route confidence {confidence:.4f} LCB {confidence_lcb:.4f} is below {floor:.2f}",
        )
    rows, rejected = _collect_rows(backend, tenant, old_dim)
    doc_ids = [doc_id for doc_id, _vector in rows]
    if rows:
        docs = np.stack([vector for _doc_id, vector in rows])
    else:
        docs = np.zeros((0, old_dim), dtype=np.float64)
    if docs.size:
        den = np.linalg.norm(docs - mu_old, axis=1)
        den = np.maximum(den, 1e-12)
        radius = _csls_radius(fit_old - mu_old, docs, mu_old, den, spec.csls_k)
    else:
        den = np.zeros((0,), dtype=np.float64)
        radius = np.zeros((0,), dtype=np.float64)
    sidecar = _pack_sidecar(den, radius)
    if len(sidecar) != spec.sidecar_bytes_per_doc * len(doc_ids):
        raise Tra1VerificationError("sidecar_bytes", "sidecar is not 8 bytes per document")
    digest = _digest(
        tenant_id=tenant,
        old_dim=old_dim,
        new_dim=new_dim,
        fit_count=fit_count,
        validation_count=val_count,
        mapping=mapping,
        mu_old=mu_old,
        mu_new=mu_new,
        confidence=confidence,
        doc_ids=doc_ids,
    )
    contract = Tra1Contract(
        tenant_id=tenant,
        old_dim=old_dim,
        new_dim=new_dim,
        rectangular=old_dim != new_dim,
        fit_count=fit_count,
        validation_count=val_count,
        confidence=confidence,
        confidence_lcb=confidence_lcb,
        sigma_min=sigma_min,
        ortho_error=ortho_error,
        sidecar_bytes_per_doc=spec.sidecar_bytes_per_doc,
        doc_count=len(doc_ids),
        digest_hex=digest,
        profile=spec.name,
        profile_version=spec.version,
        bootstrap_resamples=spec.bootstrap_resamples,
        llm_write_tokens=0,
    )
    return WrappedIndex(
        tenant_id=tenant,
        profile=spec,
        contract=contract,
        doc_ids=doc_ids,
        docs=docs,
        mu_old=mu_old,
        mu_new=mu_new,
        mapping=mapping,
        den=den,
        radius=radius,
        sidecar=sidecar,
        anchor_old=old_anchors,
        anchor_new=new_anchors,
        rejected_foreign_records=rejected,
        old_dim=old_dim,
        new_dim=new_dim,
        csls_k=spec.csls_k,
    )
