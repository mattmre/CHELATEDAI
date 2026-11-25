"""Live score-only vector adapters and an epoch-pinned D6 control plane."""
from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from chelated_grpc_gateway import (
    SCORING_BIDIR_ZEROMU_CSLS,
    SCORING_NATIVE_DOT,
    EpochPinnedEvidenceFence,
    GatewayRouteRegistration,
    ShardZeroMuCslsSidecar,
    TRA1AuthorizationError,
    compute_unseen_score_upper_bound,
)

RAW = "Uncalibrated_Raw_Merge"
ZSCORE = "DEV_ZScore_Merge"
RRF = "RRF_k60"
CSLS = "Chelated_ZeroMu_CSLS_8B"
MERGES = (RAW, ZSCORE, RRF, CSLS)
PG_HNSW = "pgvector_vector_hnsw"
PG_HALF_CTE = "pgvector_halfvec4000_hnsw_cte"
PG_EXACT_4096 = "pgvector_exact_vector4096"
OS_URL = "http://127.0.0.1:19200"
PG_DSN = "postgresql://postgres:chelated_bench_2026@127.0.0.1:15432/postgres"
EF_MAX = 512
SHORTLIST_MAX = 1024


@dataclass(frozen=True)
class VectorHit:
    doc_id: str
    point_id: int
    admitted_epoch: int
    raw_dot: float


@dataclass(frozen=True)
class SearchPage:
    hits: tuple[VectorHit, ...]
    route: str
    exact_prefix: bool


class MultiBackendVectorAdapter(Protocol):
    name: str
    dim: int
    transport: str
    fp16: bool

    def create(self) -> None: ...
    def ingest(self, doc_ids: tuple[str, ...], embeddings: np.ndarray, epoch: int) -> None: ...
    def finalize(self) -> None: ...
    def count(self) -> int: ...
    def server_info(self) -> dict: ...
    def search(
        self, query: np.ndarray, *, allowed: tuple[int, ...] | None,
        epoch: int, limit: int, ef: int, exact: bool,
    ) -> SearchPage: ...
    def close(self) -> None: ...


def _identifier(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name):
        raise ValueError(f"Invalid benchmark index identifier: {name!r}")
    return name


def _vector_rows(doc_ids: tuple[str, ...], embeddings: np.ndarray, dim: int, epoch: int) -> np.ndarray:
    rows = np.asarray(embeddings, dtype=np.float32)
    if epoch < 1 or rows.shape != (len(doc_ids), dim) or not np.isfinite(rows).all():
        raise ValueError("Invalid embedding generation")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Duplicate document IDs")
    return rows


def invert_opensearch_score(score: float) -> float:
    score = float(score)
    if not math.isfinite(score) or score <= 0.0:
        raise ValueError(f"Invalid OpenSearch innerproduct score: {score}")
    return score - 1.0 if score >= 1.0 else 1.0 - 1.0 / score


def _os_search_body(query, allowed, epoch, limit, ef, exact):
    clauses = [{"range": {"admitted_epoch": {"lte": epoch}}}]
    if allowed is not None:
        clauses.append({"terms": {"point_id": list(allowed)}})
    admission = {"bool": {"filter": clauses}}
    body = {"size": limit, "_source": {"excludes": ["embedding"]}, "track_total_hits": False}
    if exact:
        body["query"] = {"script_score": {
            "query": admission,
            "script": {"lang": "knn", "source": "knn_score", "params": {
                "field": "embedding", "query_value": query.tolist(), "space_type": "innerproduct",
            }},
        }}
        body["sort"] = [{"_score": "desc"}, {"point_id": "asc"}]
    else:
        body["query"] = {"knn": {"embedding": {
            "vector": query.tolist(), "k": limit, "filter": admission,
            "method_parameters": {"ef_search": ef},
        }}}
    return body


class LiveOpenSearchKnnBackend:
    transport = "HTTP/REST"
    fp16 = False

    def __init__(self, name: str, dim: int, url: str = OS_URL):
        self.name, self.dim = _identifier(name), dim
        self.client = httpx.Client(base_url=url, timeout=120, trust_env=False)

    def _request(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def create(self):
        self._request("PUT", f"/{self.name}", json={
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {"properties": {
                "doc_id": {"type": "keyword"},
                "point_id": {"type": "integer"},
                "admitted_epoch": {"type": "long"},
                "embedding": {"type": "knn_vector", "dimension": self.dim, "method": {
                    "name": "hnsw", "engine": "faiss", "space_type": "innerproduct",
                    "parameters": {"ef_construction": 128, "m": 16},
                }},
            }},
        })

    def ingest(self, doc_ids, embeddings, epoch):
        rows = _vector_rows(doc_ids, embeddings, self.dim, epoch)
        for start in range(0, len(doc_ids), 128):
            lines = []
            for i in range(start, min(start + 128, len(doc_ids))):
                lines.append(json.dumps({"create": {"_id": str(i)}}))
                lines.append(json.dumps({
                    "doc_id": doc_ids[i], "point_id": i, "admitted_epoch": epoch,
                    "embedding": rows[i].tolist(),
                }, separators=(",", ":")))
            reply = self._request(
                "POST", f"/{self.name}/_bulk", content=("\n".join(lines) + "\n").encode(),
                headers={"Content-Type": "application/x-ndjson"},
            )
            if reply["errors"]:
                error = next(item["create"]["error"] for item in reply["items"] if "error" in item["create"])
                raise RuntimeError(f"OpenSearch bulk ingest failed: {error}")

    def finalize(self):
        self._request("POST", f"/{self.name}/_refresh")

    def count(self):
        return int(self._request("GET", f"/{self.name}/_count")["count"])

    def server_info(self):
        return self._request("GET", "/")["version"]

    def search(self, query, *, allowed, epoch, limit, ef, exact):
        route = "opensearch_knn_score_exact" if exact else "opensearch_faiss_hnsw_requested"
        hits = []
        if limit <= 0 or allowed == ():
            return SearchPage((), route, exact)
        page_size = min(limit, 10_000 if exact else SHORTLIST_MAX)
        body = _os_search_body(query, allowed, epoch, page_size, ef, exact)
        while len(hits) < limit:
            body["size"] = min(limit - len(hits), page_size)
            answer = self._request("POST", f"/{self.name}/_search", json=body)
            if answer["timed_out"] or answer["_shards"]["failed"]:
                raise RuntimeError(f"OpenSearch returned a partial result for {self.name}")
            page = answer["hits"]["hits"]
            for hit in page:
                src = hit["_source"]
                if "embedding" in src or "fields" in hit:
                    raise RuntimeError("Unexpected OpenSearch search response fields")
                hits.append(VectorHit(
                    src["doc_id"], int(src["point_id"]), int(src["admitted_epoch"]),
                    invert_opensearch_score(hit["_score"]),
                ))
            if not exact or len(page) < body["size"]:
                break
            body["search_after"] = page[-1]["sort"]
        return SearchPage(tuple(hits), route, exact)

    def close(self):
        self.client.close()


def pgvector_index_sql(table: str, dim: int, route: str) -> str | None:
    table = _identifier(table)
    if 1 <= dim <= 2000 and route == PG_HNSW:
        expression = "embedding vector_ip_ops"
    elif dim == 4096 and route == PG_HALF_CTE:
        expression = "(subvector(embedding, 1, 4000)::halfvec(4000)) halfvec_ip_ops"
    elif dim == 4096 and route == PG_EXACT_4096:
        return None
    else:
        raise ValueError(f"Unsupported pgvector dimension/route: {dim}/{route}")
    return (f"CREATE INDEX {table}_hnsw ON {table} USING hnsw ({expression}) "
            "WITH (m = 16, ef_construction = 128)")


def pgvector_search_sql(table, route, *, query, allowed, epoch, limit, exact, k_cte=None):
    table = _identifier(table)
    if route not in (PG_HNSW, PG_HALF_CTE, PG_EXACT_4096):
        raise ValueError(f"Unknown pgvector route: {route}")
    exact = exact or route == PG_EXACT_4096
    cast = "halfvec(4096)" if route == PG_HALF_CTE else "vector"
    # The final projection never returns the vector column.
    operand = "[" + ",".join(map(str, query.tolist())) + "]"
    params = {"q": operand, "epoch": epoch, "limit": limit}
    predicate = "admitted_epoch <= %(epoch)s"
    if allowed is not None:
        predicate += " AND point_id = ANY(%(allowed)s)"
        params["allowed"] = list(allowed)
    dot = f"-1.0 * (embedding <#> %(q)s::{cast})"
    if exact:
        return (
            f"SELECT doc_id, point_id, admitted_epoch, {dot} AS raw_dot "
            f"FROM {table} WHERE {predicate} ORDER BY raw_dot DESC, point_id LIMIT %(limit)s",
            params,
        )
    if route == PG_HALF_CTE:
        params["k_cte"] = max(limit, k_cte or 4 * limit)
        return (
            "WITH shortlist AS MATERIALIZED ("
            f"SELECT doc_id, point_id, admitted_epoch, embedding FROM {table} WHERE {predicate} "
            "ORDER BY subvector(embedding, 1, 4000)::halfvec(4000) <#> "
            "subvector(%(q)s::halfvec(4096), 1, 4000)::halfvec(4000) LIMIT %(k_cte)s"
            f") SELECT doc_id, point_id, admitted_epoch, {dot} AS raw_dot FROM shortlist "
            "ORDER BY raw_dot DESC, point_id LIMIT %(limit)s",
            params,
        )
    return (
        "WITH shortlist AS MATERIALIZED ("
        f"SELECT doc_id, point_id, admitted_epoch, {dot} AS raw_dot FROM {table} WHERE {predicate} "
        "ORDER BY embedding <#> %(q)s::vector LIMIT %(limit)s"
        ") SELECT doc_id, point_id, admitted_epoch, raw_dot FROM shortlist "
        "ORDER BY raw_dot DESC, point_id",
        params,
    )



class LivePgvectorBackend:
    transport = "PostgreSQL wire protocol/TCP"
    _EXACT_SETTINGS = (
        "SET LOCAL enable_indexscan = off",
        "SET LOCAL enable_indexonlyscan = off",
        "SET LOCAL enable_bitmapscan = off",
    )

    def __init__(self, name: str, dim: int, *, route: str | None = None, dsn: str = PG_DSN):
        import psycopg2

        self.name, self.dim = _identifier(name), dim
        self.route = route or (PG_HALF_CTE if dim == 4096 else PG_HNSW)
        self.index_sql = pgvector_index_sql(self.name, dim, self.route)
        self.fp16 = self.route == PG_HALF_CTE
        self.conn = psycopg2.connect(
            dsn, connect_timeout=5, options="-c statement_timeout=600000 -c work_mem=256MB",
        )
        self.conn.autocommit = True
        self._lock = threading.RLock()
        self.iterative_scan = False

    def create(self):
        from pgvector.psycopg2 import register_vector

        with self._lock, self.conn, self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            version_pair = re.match(r"^(\d+)\.(\d+)", cur.fetchone()[0])
            self.iterative_scan = bool(version_pair and tuple(map(int, version_pair.groups())) >= (0, 8))
            cast = "halfvec" if self.fp16 else "vector"
            cur.execute(
                f"CREATE TABLE {self.name} (doc_id text NOT NULL, point_id integer NOT NULL, "
                f"admitted_epoch bigint NOT NULL, embedding {cast}({self.dim}) NOT NULL)"
            )
            register_vector(self.conn)

    def ingest(self, doc_ids, embeddings, epoch):
        from psycopg2.extras import execute_values

        rows = _vector_rows(doc_ids, embeddings, self.dim, epoch)
        cast = "halfvec(4096)" if self.fp16 else "vector"
        with self._lock, self.conn, self.conn.cursor() as cur:
            execute_values(
                cur, f"INSERT INTO {self.name} (doc_id, point_id, admitted_epoch, embedding) VALUES %s",
                ((did, i, epoch, rows[i]) for i, did in enumerate(doc_ids)),
                template=f"(%s, %s, %s, %s::{cast})", page_size=128,
            )

    def finalize(self):
        with self._lock, self.conn, self.conn.cursor() as cur:
            if self.index_sql is not None:
                cur.execute("SET LOCAL maintenance_work_mem = '512MB'")
                cur.execute(self.index_sql)
            cur.execute(f"ANALYZE {self.name}")

    def count(self):
        with self._lock, self.conn, self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.name}")
            return int(cur.fetchone()[0])

    def server_info(self):
        with self._lock, self.conn, self.conn.cursor() as cur:
            cur.execute("SELECT version(), default_version, installed_version "
                        "FROM pg_available_extensions WHERE name = 'vector'")
            server, default, installed = cur.fetchone()
        return {"server": server, "available_extension": default, "installed_extension": installed,
                "iterative_scan": self.iterative_scan}

    def _query(self, query, *, allowed, epoch, limit, ef, exact, explain=False):
        exact = exact or self.route == PG_EXACT_4096
        sql, params = pgvector_search_sql(
            self.name, self.route, query=query, allowed=allowed, epoch=epoch, limit=limit, exact=exact,
        )
        if exact:
            settings = self._EXACT_SETTINGS
        else:
            settings = ["SET LOCAL enable_seqscan = off", "SET LOCAL hnsw.ef_search = %(ef_search)s"]
            params["ef_search"] = ef
            if self.iterative_scan:
                settings.extend((
                    "SET LOCAL hnsw.iterative_scan = 'relaxed_order'",
                    "SET LOCAL hnsw.max_scan_tuples = 200000",
                    "SET LOCAL hnsw.scan_mem_multiplier = 8",
                ))
        statement = ("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " if explain else "") + sql
        # In autocommit, PostgreSQL executes one multi-statement simple-query message
        # in an implicit transaction. SET LOCAL applies to the SELECT in that message.
        with self._lock, self.conn.cursor() as cur:
            cur.execute(";\n".join((*settings, statement)), params)
            return cur.fetchall()

    def explain(self, query, **kwargs):
        return "\n".join(line[0] for line in self._query(query, explain=True, **kwargs))

    def search(self, query, *, allowed, epoch, limit, ef, exact):
        exact = exact or self.route == PG_EXACT_4096
        route = self.route
        if exact and route != PG_EXACT_4096:
            route = "pgvector_halfvec4096_exact_fallback" if self.fp16 else "pgvector_vector_exact_fallback"
        if limit <= 0 or allowed == ():
            return SearchPage((), route, exact)
        rows = self._query(query, allowed=allowed, epoch=epoch, limit=limit, ef=ef, exact=exact)
        return SearchPage(tuple(VectorHit(d, p, e, float(score)) for d, p, e, score in rows), route, exact)

    def close(self):
        self.conn.close()


class LiveQdrantBackend:
    transport = "Qdrant gRPC (search/count); HTTP telemetry"
    fp16 = False

    def __init__(self, name: str, dim: int, host="127.0.0.1", port=16333, grpc_port=16334):
        from qdrant_client import QdrantClient, models

        self.name, self.dim = _identifier(name), dim
        self.client = QdrantClient(host=host, port=port, grpc_port=grpc_port, prefer_grpc=True, timeout=120)
        self.models = models
        self.http_url = f"http://{host}:{port}"
        self.doc_ids: tuple[str, ...] = ()
        self.epoch = 0

    def create(self):
        m = self.models
        self.client.create_collection(
            collection_name=self.name,
            vectors_config=m.VectorParams(size=self.dim, distance=m.Distance.DOT),
            hnsw_config=m.HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10),
            optimizers_config=m.OptimizersConfigDiff(indexing_threshold=0, default_segment_number=1),
            shard_number=1,
        )

    def ingest(self, doc_ids, embeddings, epoch):
        rows = _vector_rows(doc_ids, embeddings, self.dim, epoch)
        for start in range(0, len(doc_ids), 128):
            end = min(start + 128, len(doc_ids))
            self.client.upsert(
                collection_name=self.name, wait=True,
                points=self.models.Batch(ids=list(range(start, end)), vectors=rows[start:end].tolist()),
            )
        self.doc_ids, self.epoch = doc_ids, epoch

    def finalize(self):
        self.client.update_collection(
            collection_name=self.name,
            optimizers_config=self.models.OptimizersConfigDiff(indexing_threshold=1),
        )
        deadline = time.monotonic() + 900
        while True:
            info = self.client.get_collection(self.name)
            if info.status == self.models.CollectionStatus.GREEN and info.indexed_vectors_count == len(self.doc_ids):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Qdrant did not index all points for {self.name}: {info}")
            time.sleep(2)

    def count(self):
        return int(self.client.count(self.name, exact=True).count)

    def index_info(self):
        info = self.client.get_collection(self.name)
        return {"points_count": info.points_count, "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value}

    def server_info(self):
        with httpx.Client(base_url=self.http_url, timeout=15, trust_env=False) as client:
            response = client.get("/telemetry", params={"details_level": 0})
            response.raise_for_status()
            telemetry = response.json()["result"]
        return telemetry["app"]

    def search(self, query, *, allowed, epoch, limit, ef, exact):
        route = "qdrant_dot_exact" if exact else "qdrant_dot_ann_requested"
        if limit <= 0 or allowed == () or epoch < self.epoch:
            return SearchPage((), route, exact)
        m = self.models
        admission = m.Filter(must=[m.HasIdCondition(has_id=list(allowed))]) if allowed is not None else None
        points = self.client.query_points(
            collection_name=self.name, query=query.tolist(), query_filter=admission, limit=limit,
            search_params=m.SearchParams(
                hnsw_ef=ef, exact=exact, quantization=m.QuantizationSearchParams(ignore=True) if exact else None,
            ),
            with_vectors=False, with_payload=False,
        ).points
        hits = []
        for point in points:
            if point.vector or point.payload:
                raise RuntimeError("Unexpected Qdrant vector or payload egress")
            idx = int(point.id)
            if not 0 <= idx < len(self.doc_ids):
                raise RuntimeError(f"Unknown Qdrant point ID: {idx}")
            hits.append(VectorHit(self.doc_ids[idx], idx, self.epoch, float(point.score)))
        return SearchPage(tuple(hits), route, exact)

    def close(self):
        self.client.close()


def verify_backend_dot_inversion(backend: MultiBackendVectorAdapter, *, exact_modes=(False, True)) -> dict:
    """Gate live recovered scores against float64 references for original and stored operands."""
    rng = np.random.default_rng(1106)
    n = 260
    groups = {
        "negative": rng.uniform(-0.85, -0.01, n),
        "zero": np.zeros(n),
        "positive": rng.uniform(0.01, 0.85, n),
        "near_tie": np.linspace(0.3275, 0.327504, n),
    }
    dim = backend.dim
    if dim < 3:
        raise ValueError("Score canaries require at least three dimensions")
    query = np.full(dim, 1 / math.sqrt(dim), dtype=np.float32)
    q = query.astype(np.float64)
    all_dots = np.concatenate(list(groups.values()))
    orthogonal = rng.normal(size=(len(all_dots), dim))
    orthogonal -= ((orthogonal @ q) / (q @ q))[:, None] * q
    orthogonal /= np.linalg.norm(orthogonal, axis=1, keepdims=True)
    docs = (all_dots[:, None] * q + np.sqrt(1 - all_dots[:, None] ** 2) * orthogonal).astype(np.float32)
    doc_ids = tuple(f"canary_{i}" for i in range(len(docs)))
    backend.create()
    backend.ingest(doc_ids, docs, 1)
    backend.finalize()
    post_ingest_count = backend.count()
    if post_ingest_count != len(doc_ids):
        raise RuntimeError(f"Canary ingest count mismatch for {backend.name}")

    original = docs.astype(np.float64) @ q
    stored = (docs.astype(np.float16).astype(np.float64) @ query.astype(np.float16).astype(np.float64)
              if backend.fp16 else original)
    precision_limit = 2e-3 if backend.fp16 else 1e-5
    modes = {}
    for exact in exact_modes:
        errors = {}
        routes = set()
        for group_index, group in enumerate(groups):
            start = group_index * n
            stop = start + n
            recovered = {}
            for offset in range(start, stop, 52):
                allowed = tuple(range(offset, offset + 52))
                page = backend.search(query, allowed=allowed, epoch=1, limit=len(allowed), ef=EF_MAX, exact=exact)
                routes.add(page.route)
                min_expected = len(allowed) if (exact or page.exact_prefix) else 1
                returned_ids = [hit.point_id for hit in page.hits]
                if (
                    len(set(returned_ids)) != len(returned_ids)
                    or not (set(returned_ids) <= set(allowed))
                    or len(returned_ids) < min_expected
                    or any(hit.doc_id != doc_ids[hit.point_id] or hit.admitted_epoch != 1 for hit in page.hits)
                ):
                    raise RuntimeError(f"Incomplete canary coverage for {backend.name}/{group}/{page.route}")
                recovered.update((hit.point_id, hit.raw_dot) for hit in page.hits)
            checked_ids = sorted(recovered)
            if not checked_ids:
                raise RuntimeError(f"Zero canary hits recovered for {backend.name}/{group}")
            observed = np.array([recovered[i] for i in checked_ids])
            orig_sub = original[checked_ids]
            stor_sub = stored[checked_ids]
            err = float(np.max(np.abs(observed - orig_sub)))
            arithmetic = float(np.max(np.abs(observed - stor_sub)))
            rounding = float(np.max(np.abs(stor_sub - orig_sub)))
            if not (err < precision_limit and arithmetic < 1e-5):
                raise RuntimeError(f"Dot canary failed for {backend.name}/{group}: {err=}, {arithmetic=}")
            errors[group] = {"hits": len(checked_ids), "max_abs_err": err, "max_abs_err_against_stored_fp64": arithmetic,
                             "max_abs_fp16_rounding_err": rounding}
        modes["exact" if exact else "ann_requested"] = {
            "hits_checked": sum(group["hits"] for group in errors.values()), "routes": sorted(routes),
            "groups": errors, "max_abs_err": max(group["max_abs_err"] for group in errors.values()),
        }
    return {"post_ingest_count": post_ingest_count, "modes": modes,
            "max_abs_err_limit": precision_limit, "stored_operand_error_limit": 1e-5}



def shortlist_policy(k_os: int, p_admit: float) -> tuple[int, int, bool]:
    if not 1 <= k_os <= SHORTLIST_MAX or not 0.0 <= p_admit <= 1.0:
        raise ValueError("Invalid shortlist policy input")
    p_floor = max(p_admit, 0.05)
    return min(SHORTLIST_MAX, math.ceil(4 * k_os / p_floor)), min(EF_MAX, math.ceil(2 * k_os / p_floor)), p_admit <= 0.15


@dataclass(frozen=True)
class RouteClaims:
    query_space_id: str
    bridge_contract_id: str
    weights_sha256: str
    index_generation: str


@dataclass(frozen=True)
class VectorShard:
    backend: MultiBackendVectorAdapter
    sidecar: ShardZeroMuCslsSidecar
    dev_mean: float
    dev_std: float

    def __post_init__(self):
        if (
            self.backend.dim != self.sidecar.dim or self.backend.name != self.sidecar.collection_name
            or not self.sidecar.n_docs or len(set(self.sidecar.doc_ids)) != self.sidecar.n_docs
            or not math.isfinite(self.dev_mean) or not math.isfinite(self.dev_std) or self.dev_std <= 0
        ):
            raise ValueError("Backend generation, DEV calibration and sidecar do not align")


@dataclass(frozen=True)
class ShardSearch:
    candidates: tuple[VectorHit, ...]
    initial: tuple[VectorHit, ...]
    route: str
    initial_route: str | None
    exact_fallback: bool
    exhausted: bool
    refills: int
    raw_stale_hits: int
    raw_hits: int
    ann_request_rtt_ms: float


@dataclass(frozen=True)
class ApplianceSearch:
    epoch: int
    merges: dict[str, list[str]]
    initial_merges: dict[str, list[str]]
    shards: tuple[ShardSearch, ...]


def _filtered_hits(page, doc_ids, allowed, epoch):
    hits = {}
    stale = 0
    for h in page.hits:
        if (
            h.point_id not in allowed or h.doc_id != doc_ids[h.point_id]
            or not 1 <= h.admitted_epoch <= epoch
        ):
            stale += 1
            continue
        if h.point_id in hits or not math.isfinite(h.raw_dot):
            raise RuntimeError(f"Duplicate or invalid score in {page.route}")
        hits[h.point_id] = h
    return tuple(sorted(hits.values(), key=lambda h: (-h.raw_dot, h.doc_id))), stale


def _centered(shard, query, hits):
    idx = np.fromiter((h.point_id for h in hits), dtype=np.intp)
    dots = np.fromiter((h.raw_dot for h in hits), dtype=np.float64)
    sidecar = shard.sidecar
    shift = float(query.astype(np.float64) @ sidecar.mu_d)
    scores = (dots - shift) / sidecar.doc_centered_norms[idx] if sidecar.zero_mu_active else dots
    return scores, sidecar.csls_r_d[idx], shift


def _csls_scores(shard, query, hits):
    z, r_d, _ = _centered(shard, query, hits)
    kq = min(shard.sidecar.csls_k, len(z))
    r_q = float(np.sort(z)[-kq:].mean()) if kq else 0.0
    return 2 * z - r_d - r_q


def _certify_prefix(shard, query, hits, admitted, limit, top_k):
    """Prove the CSLS query penalty and final top-k from an exact dot prefix."""
    if len(hits) == len(admitted):
        return True, True
    sidecar = shard.sidecar
    kq = min(sidecar.csls_k, len(admitted))
    if len(hits) != limit or len(hits) < max(top_k, kq) or hits[top_k - 1].raw_dot <= hits[-1].raw_dot:
        return False, False
    z, penalties, shift = _centered(shard, query, hits)
    unseen = np.fromiter(admitted.difference(h.point_id for h in hits), dtype=np.intp)
    norms = sidecar.doc_centered_norms[unseen]
    r_min = float(np.min(sidecar.csls_r_d[unseen]))
    z_bound = compute_unseen_score_upper_bound(
        dot_upper_bound=hits[-1].raw_dot, c_shift=shift,
        n_min=float(np.min(norms)), n_max=float(np.max(norms)), use_csls=False,
    )
    rq_top = np.sort(z)[-kq:]
    if rq_top[0] <= z_bound:
        return False, False
    # r_q is constant within a shard and cancels from the ranking proof.
    kth = np.sort(2 * z - penalties)[-top_k]
    return bool(kth > 2 * z_bound - r_min), False


def _search_shard(shard, query, snapshot, *, top_k, k_os, force_exhaust=False):
    doc_ids = shard.sidecar.doc_ids
    allowed = tuple(i for i, did in enumerate(doc_ids) if did in snapshot.admitted_doc_ids)
    admitted = frozenset(allowed)
    if not allowed:
        return ShardSearch((), (), "empty_admission", None, False, True, 0, 0, 0, 0.0)
    depth, ef, low_admit = shortlist_policy(k_os, len(allowed) / len(doc_ids))
    depth = min(max(depth, top_k), len(allowed))
    wire_allowed = None if len(allowed) == len(doc_ids) else allowed
    stale_total = raw_total = 0

    def fetch(limit, exact):
        nonlocal stale_total, raw_total
        page = shard.backend.search(
            query, allowed=wire_allowed, epoch=snapshot.epoch, limit=limit, ef=ef, exact=exact,
        )
        if len(page.hits) > limit or (exact and not page.exact_prefix):
            raise RuntimeError(f"Backend violated search contract: {page.route}")
        rows, stale = _filtered_hits(page, doc_ids, admitted, snapshot.epoch)
        stale_total += stale
        raw_total += len(page.hits)
        return rows, page

    initial, initial_route, ann_rtt = (), None, 0.0
    if not (force_exhaust or low_admit):
        start = time.perf_counter()
        initial, page = fetch(depth, exact=False)
        ann_rtt = (time.perf_counter() - start) * 1000
        initial_route = page.route
        if len(initial) == len(allowed):
            return ShardSearch(initial, initial, page.route, initial_route, False, True, 0,
                               stale_total, raw_total, ann_rtt)
        if page.exact_prefix and len(initial) == len(page.hits):
            certified, exhausted = _certify_prefix(shard, query, initial, admitted, depth, top_k)
            if certified:
                return ShardSearch(initial, initial, page.route, initial_route, False, exhausted, 0,
                                   stale_total, raw_total, ann_rtt)

    limit = len(allowed) if force_exhaust or low_admit else depth
    refills = 0
    while True:
        rows, page = fetch(limit, exact=True)
        # Keep exact-prefix certification separate from a filtered or short response.
        certified, exhausted = (False, False)
        if len(rows) == len(page.hits):
            certified, exhausted = _certify_prefix(shard, query, rows, admitted, limit, top_k)
        if certified:
            return ShardSearch(rows, initial, page.route, initial_route, not force_exhaust, exhausted, refills,
                               stale_total, raw_total, ann_rtt)
        if limit == len(allowed):
            raise RuntimeError(f"Cannot verify admitted exhaustion on {shard.backend.name} at epoch {snapshot.epoch}")
        limit = min(len(allowed), 2 * limit, SHORTLIST_MAX) if limit < SHORTLIST_MAX else len(allowed)
        refills += 1


def _merge(shards, queries, hit_groups, top_k):
    values: dict[str, dict[str, float]] = {mode: {} for mode in MERGES}
    for shard, query, hits in zip(shards, queries, hit_groups, strict=True):
        for rank, (hit, csls) in enumerate(zip(hits, _csls_scores(shard, query, hits), strict=True), 1):
            did, dot = hit.doc_id, hit.raw_dot
            values[RAW][did] = dot
            values[ZSCORE][did] = (dot - shard.dev_mean) / shard.dev_std
            values[RRF][did] = values[RRF].get(did, 0.0) + 1.0 / (60 + rank)
            values[CSLS][did] = float(csls)
    return {
        mode: sorted(scores, key=lambda did: (-scores[did], did))[:top_k]
        for mode, scores in values.items()
    }


class ChelatedControlPlane:
    """One TRA-1 route and one EK5 snapshot across two disjoint index generations."""

    def __init__(self, legacy: VectorShard, native: VectorShard, registration: GatewayRouteRegistration,
                 fence: EpochPinnedEvidenceFence):
        self.shards = (legacy, native)
        sides = (legacy.sidecar, native.sidecar)
        if registration.primary_sidecar is sides[0]:
            input_shard, target = native, legacy
        else:
            input_shard, target = legacy, native
        if (
            registration.primary_sidecar is not target.sidecar
            or registration.secondary_sidecar is not input_shard.sidecar
            or registration.bridge is None or registration.bridge_contract is None
            or registration.resident_index.document_count != target.sidecar.n_docs
            or registration.query_space.dimension != input_shard.backend.dim
            or registration.resident_index.document_space.dimension != target.backend.dim
            or registration.resident_index.index_id != target.backend.name
            or set(legacy.sidecar.doc_ids).intersection(native.sidecar.doc_ids)
        ):
            raise TRA1AuthorizationError("D6 route does not bind the selected two-shard generation")
        self.registration = registration
        self.fence = fence
        self._legacy_is_input = input_shard is legacy

    def prepare_query(self, query, claims):
        reg = self.registration
        reg.verify_tra1_ingress(
            claimed_query_space_id=claims.query_space_id,
            claimed_bridge_contract_id=claims.bridge_contract_id,
            claimed_weights_sha256=claims.weights_sha256,
            claimed_index_generation=claims.index_generation,
        )
        direct, _ = reg.project_query_causal(query, scoring_mode=SCORING_NATIVE_DOT, update_ema=False)
        bridged, _ = reg.project_query_causal(query, scoring_mode=SCORING_BIDIR_ZEROMU_CSLS, update_ema=False)
        return (direct, bridged) if self._legacy_is_input else (bridged, direct)

    def search(self, query: np.ndarray, claims: RouteClaims, *, epoch: int, top_k: int = 10,
               k_os: int = 64, wait_for_epoch: bool = True, force_exhaust: bool = False) -> ApplianceSearch:
        if epoch < 1 or not 1 <= top_k <= k_os <= SHORTLIST_MAX:
            raise TRA1AuthorizationError("A positive epoch and valid search limits are required")
        queries = self.prepare_query(query, claims)
        snapshot = self.fence.wait_and_get_snapshot(epoch, wait_for_epoch=wait_for_epoch)
        if snapshot.epoch != epoch:
            raise TRA1AuthorizationError("The requested epoch has no exact committed snapshot")
        results = tuple(
            _search_shard(shard, q, snapshot, top_k=top_k, k_os=k_os, force_exhaust=force_exhaust)
            for shard, q in zip(self.shards, queries, strict=True)
        )
        return ApplianceSearch(
            snapshot.epoch,
            _merge(self.shards, queries, (r.candidates for r in results), top_k),
            _merge(self.shards, queries, (r.initial for r in results), top_k),
            results,
        )

