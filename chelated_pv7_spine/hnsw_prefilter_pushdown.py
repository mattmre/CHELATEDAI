
"""Filtered HNSW adapters and execution evidence for Bundle 48."""

from __future__ import annotations

import heapq
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass

import httpx
import numpy as np

from chelated_multibackend_appliance import (
    PG_HNSW,
    LiveOpenSearchKnnBackend,
    _identifier,
    _os_search_body,
    pgvector_search_sql,
)

TOP_K = 10
BASE_EF = 220
EF_CAP = 1000
TIMEOUT_SECONDS = 5
_PINNED_VERSIONS = {"qdrant": "1.20", "opensearch": "3.9", "pgvector": "0.8.7"}


def require_version(backend, version):
    pin = _PINNED_VERSIONS[backend]
    pattern = re.escape(pin) if backend == "pgvector" else re.escape(pin) + r"\.\d+"
    if not isinstance(version, str) or not re.fullmatch(pattern, version):
        raise RuntimeError(f"{backend} requires {pin}, got {version!r}")


def adaptive_ef(admitted_frac, *, base_ef=BASE_EF, ef_cap=EF_CAP):
    if not 0 <= admitted_frac <= 1 or not 1 <= base_ef <= ef_cap:
        raise ValueError("Invalid admission fraction or HNSW search budget")
    return min(ef_cap, math.ceil(base_ef / max(admitted_frac, 0.02)))


@dataclass(frozen=True)
class Admission:
    epoch: int
    admitted_count: int
    population: int
    allowed: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.epoch < 1 or not 0 <= self.admitted_count <= self.population or self.population < 1:
            raise ValueError("Invalid admission snapshot")

    @property
    def ef(self):
        return adaptive_ef(self.admitted_count / self.population)


@dataclass(frozen=True)
class SearchResult:
    ids: tuple[int, ...]
    visited_graph_candidates: int | None = None


def qdrant_filter(admission):
    """Native REST serialization of Filter(must=[FieldCondition, HasIdCondition?])."""
    must = [{"key": "admitted_epoch", "range": {"lte": admission.epoch}}]
    if admission.allowed is not None:
        must.append({"has_id": list(admission.allowed)})
    return {"must": must}


def _nodes(value):
    """Walk backend-supplied tree-shaped telemetry."""
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nodes(child)


def _counter(value):
    if isinstance(value, Mapping):
        value = value.get("count")
    return value if type(value) is int and value >= 0 else None


def qdrant_graph_visits(payload):
    """Use the direct graph visit counter when server instrumentation exposes it.

    Stock Qdrant search route counters count calls. Response CPU units count
    work and cannot establish how many graph candidates were visited.
    """
    for container in (payload, payload.get("usage", {}), payload.get("result", {})):
        if isinstance(container, Mapping):
            count = _counter(container.get("visited_graph_candidates"))
            if count is not None:
                return count
    return None


# Qdrant 1.20's dispatch.rs uses small_cardinality for a plain index scan.
_QDRANT_ROUTES = (
    "filtered_plain", "filtered_small_cardinality", "filtered_large_cardinality",
    "filtered_acorn", "filtered_exact",
)


def qdrant_route_snapshot(payload, collection):
    """Sum segment vector-index counters for only the dedicated collection."""
    root = payload.get("result", payload)
    if not isinstance(root, Mapping):
        return None
    matches = [
        node for node in _nodes(root.get("collections", {}))
        if node.get("id") == collection or node.get("name") == collection
    ]
    totals = dict.fromkeys(_QDRANT_ROUTES, 0)
    indexes = 0
    for match in matches:
        for node in _nodes(match):
            searches = node.get("vector_index_searches")
            if not isinstance(searches, list) or node.get("info", {}).get("num_points") == 0:
                continue
            for search in searches:
                counts = {route: _counter(search.get(route, 0)) for route in _QDRANT_ROUTES}
                if any(count is None for count in counts.values()):
                    return None
                indexes += 1
                for route, count in counts.items():
                    totals[route] += count
    return {"indexes": indexes, **totals} if indexes else None


def qdrant_route_evidence(before, after, expected_searches, *, exact_controls=0):
    if (
        before is None or after is None or before["indexes"] != after["indexes"]
        or any(after[key] < before[key] for key in _QDRANT_ROUTES)
    ):
        return {"route": "unknown", "exact_fallback_triggered": None, "route_counts": None}
    delta = {key: after[key] - before[key] for key in _QDRANT_ROUTES}
    graph = delta["filtered_large_cardinality"] + delta["filtered_acorn"]
    flat = delta["filtered_plain"] + delta["filtered_small_cardinality"]
    expected_exact = exact_controls * after["indexes"]
    expected_graph = expected_searches * after["indexes"]
    if flat or delta["filtered_exact"] > expected_exact:
        route, fallback = "filtered_plain", True
    elif graph == expected_graph and delta["filtered_exact"] == expected_exact:
        route, fallback = "filtered_hnsw", False
    else:
        route, fallback = "unknown", None
    return {"route": route, "exact_fallback_triggered": fallback, "indexed_segments": after["indexes"],
            "route_counts": delta}


def parse_opensearch_plan(profile, explain):
    """Require actual native ANN and filter-bitset execution metrics from the profile."""
    descriptions = [
        node["description"] for node in _nodes(explain)
        if isinstance(node.get("description"), str)
    ]
    leaf_prefix = "the type of knn search executed at leaf was "
    leaf = [value.lower().removeprefix(leaf_prefix) for value in descriptions if value.lower().startswith(leaf_prefix)]
    nodes = [
        node for node in _nodes(profile.get("profile", {})) if isinstance(node.get("type"), str)
    ]
    native = [
        item for node in nodes if node["type"].lower() in {"knnquery", "nativeengineknnvectorquery"}
        for item in _nodes(node)
    ]
    keys = ("ann_search_count", "exact_search_count", "bitset_creation_count", "cardinality")
    counters = {
        key: max((count for item in native if (count := _counter(item.get(key))) is not None), default=None)
        for key in keys
    }
    executed_exact = bool(counters["exact_search_count"]) or any(value.startswith("exact") for value in leaf)
    prefilter_ann = (
        bool(counters["ann_search_count"] and counters["bitset_creation_count"] and counters["cardinality"])
        and counters["exact_search_count"] == 0
    )
    return {
        "route": "exact" if executed_exact else "filtered_hnsw" if prefilter_ann else "unknown",
        "exact_fallback_triggered": True if executed_exact else False if prefilter_ann else None,
        "profile_node_types": sorted({node["type"] for node in nodes}),
        "profile_counters": counters, "explain_descriptions": descriptions,
    }


def parse_pgvector_plan(payload, *, hnsw_index, iterative_scan, expected_hits):
    """Use the known HNSW index and session mode. EXPLAIN omits HNSW resume counts."""
    root = payload[0]["Plan"]
    nodes = [node for node in _nodes(root) if "Node Type" in node and node.get("Actual Loops", 0) > 0]
    indexes = [node for node in nodes if node["Node Type"] == "Index Scan" and node.get("Index Name") == hnsw_index]
    scans = [node for node in nodes if node.get("Relation Name") and "Scan" in node["Node Type"]]
    exact = [node for node in scans if node not in indexes]
    scan = (indexes or scans or [root])[0]
    route = "unknown"
    if exact:
        route = "bitmap_scan" if any(node["Node Type"] == "Bitmap Heap Scan" for node in exact) else "exact_scan"
    elif indexes:
        route = "iterative_hnsw" if iterative_scan == "relaxed_order" else "hnsw"
    return {
        "route": route,
        "plan_node_type": scan["Node Type"],
        "index_name": scan.get("Index Name"),
        "iterative_scan": iterative_scan,
        "shared_hit_blocks": int(root.get("Shared Hit Blocks", 0)),
        "rows_removed_by_filter": sum(
            int(node.get("Rows Removed by Filter", 0) * node["Actual Loops"]) for node in scans
        ),
        "short_results": max(0, expected_hits - int(root["Actual Rows"])),
        "exact_fallback_count": int(bool(exact)) if scans else None,
        "exact_fallback_triggered": bool(exact) if scans else None,
    }


class SimulatedHnsw:
    """Seeded, single-layer graph traversal with an integer admission bitset."""

    name = "simulated_hnsw"

    def __init__(self, vectors, epochs):
        self.vectors = vectors
        self.epochs = epochs
        size = len(vectors)
        similarity = vectors @ vectors.T
        np.fill_diagonal(similarity, -np.inf)
        nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :16]
        # Ring links keep the graph connected independently of the vector distribution.
        self.graph = [
            tuple(dict.fromkeys((*map(int, nearest[i]), (i - 1) % size, (i + 1) % size)))
            for i in range(size)
        ]

    def search(self, query, admission, *, exact=False, limit=TOP_K):
        permitted = np.flatnonzero(self.epochs <= admission.epoch)
        if admission.allowed is not None:
            permitted = np.intersect1d(permitted, admission.allowed)
        if exact:
            scores = self.vectors[permitted] @ query
            order = np.lexsort((permitted, -scores))
            return SearchResult(tuple(map(int, permitted[order[:limit]])))
        if not len(permitted):
            return SearchResult((), 0)
        eligible = sum(1 << int(i) for i in permitted)
        frontier = []
        visited = set()
        hits = []

        def visit(point):
            visited.add(point)
            heapq.heappush(frontier, (-float(self.vectors[point] @ query), point))

        budget = max(admission.ef, limit)
        visit(0)
        while frontier:
            score, point = heapq.heappop(frontier)
            if eligible & (1 << point):
                hits.append((score, point))
            for neighbor in self.graph[point]:
                if len(visited) >= budget:
                    break
                if neighbor not in visited:
                    visit(neighbor)
        return SearchResult(tuple(point for _, point in sorted(hits)[:limit]), len(visited))

    def profile(self, query, admission):
        probe = self.search(query, admission)
        return {"route": "simulated_filtered_graph", "exact_fallback_triggered": False,
                "visited_graph_candidates": probe.visited_graph_candidates}


class QdrantPrefilter:
    name = "qdrant"

    def __init__(self, collection, url, *, api_key=None):
        self.collection = _identifier(collection)
        self.path = f"/collections/{self.collection}"
        headers = {"api-key": api_key} if api_key else None
        self.client = httpx.Client(base_url=url, timeout=TIMEOUT_SECONDS, trust_env=False, headers=headers)

    def _request(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        answer = response.json()
        if answer.get("status") not in ("ok", None):
            raise RuntimeError("Qdrant returned an unsuccessful response")
        return answer

    def server_info(self):
        version = self._request("GET", "/")["version"]
        require_version(self.name, version)
        return {"version": version}

    def create(self, dim):
        self._request("PUT", self.path, timeout=120, json={
            "vectors": {"size": dim, "distance": "Dot"},
            "hnsw_config": {"m": 64, "ef_construct": 512, "full_scan_threshold": 10},
            "optimizers_config": {"indexing_threshold": 0, "default_segment_number": 1},
            "shard_number": 1,
        })

    def ingest(self, vectors, epochs):
        self._request("PUT", f"{self.path}/index", params={"wait": "true"}, timeout=120, json={
            "field_name": "admitted_epoch", "field_schema": "integer",
        })
        for start in range(0, len(vectors), 256):
            stop = min(start + 256, len(vectors))
            self._request("PUT", f"{self.path}/points", params={"wait": "true"}, timeout=120, json={
                "batch": {
                    "ids": list(range(start, stop)),
                    "vectors": vectors[start:stop].tolist(),
                    "payloads": [{"admitted_epoch": int(epoch)} for epoch in epochs[start:stop]],
                },
            })
        self._request("PATCH", self.path, timeout=120, json={"optimizers_config": {"indexing_threshold": 1}})
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            info = self._request("GET", self.path, timeout=30)["result"]
            if info["status"] == "green" and info["indexed_vectors_count"] == len(vectors):
                return
            time.sleep(1)
        raise TimeoutError("Qdrant fixture did not finish building its HNSW index")

    def search(self, query, admission, *, exact=False, limit=TOP_K):
        if not admission.admitted_count:
            return SearchResult(())
        response = self._request("POST", f"{self.path}/points/query", json={
            "query": query.tolist(), "filter": qdrant_filter(admission), "limit": limit,
            "params": {"hnsw_ef": admission.ef, "exact": exact},
            "with_payload": False, "with_vector": False,
        })
        return SearchResult(
            tuple(int(point["id"]) for point in response["result"]["points"]),
            qdrant_graph_visits(response) if not exact else None,
        )

    def route_snapshot(self):
        try:
            telemetry = self._request("GET", "/telemetry", params={"details_level": 10}, timeout=30)
        except httpx.HTTPError:
            return None
        return qdrant_route_snapshot(telemetry, self.collection)

    def profile(self, query, admission):
        before = self.route_snapshot()
        result = self.search(query, admission)
        evidence = qdrant_route_evidence(before, self.route_snapshot(), 1)
        return {**evidence, "visited_graph_candidates": result.visited_graph_candidates}

    def drop(self):
        self._request("DELETE", self.path)

    def close(self):
        self.client.close()


class OpenSearchPrefilter(LiveOpenSearchKnnBackend):
    """FAISS efficient filtering on the D6 index format."""

    engine = "faiss"

    def __init__(self, name, dim, url):
        self.name, self.dim = _identifier(name), dim
        self.client = httpx.Client(base_url=url, timeout=TIMEOUT_SECONDS, trust_env=False)

    def server_info(self):
        version = super().server_info()["number"]
        require_version("opensearch", version)
        return {"version": version, "engine": self.engine}

    def create(self):
        self._request("PUT", f"/{self.name}", timeout=120, json={
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {"properties": {
                "doc_id": {"type": "keyword"},
                "point_id": {"type": "integer"},
                "admitted_epoch": {"type": "long"},
                "embedding": {"type": "knn_vector", "dimension": self.dim, "method": {
                    "name": "hnsw", "engine": "faiss", "space_type": "innerproduct",
                    "parameters": {"ef_construction": 512, "m": 64},
                }},
            }},
        })

    def ingest_fixture(self, vectors, epochs):
        self._request("PUT", f"/{self.name}/_settings", timeout=60, json={
            "index.knn.advanced.filtered_exact_search_threshold": 0,
            "index.knn.faiss.efficient_filter.disable_exact_search": True,
        })
        for start in range(0, len(vectors), 256):
            lines = []
            for point in range(start, min(start + 256, len(vectors))):
                lines.append(json.dumps({"create": {"_id": str(point)}}))
                lines.append(json.dumps({
                    "doc_id": str(point), "point_id": point,
                    "admitted_epoch": int(epochs[point]), "embedding": vectors[point].tolist(),
                }))
            result = self._request(
                "POST", f"/{self.name}/_bulk", content="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"}, timeout=120,
            )
            if result["errors"]:
                raise RuntimeError("OpenSearch fixture bulk ingest failed")
        self._request("POST", f"/{self.name}/_refresh", timeout=120)
        self._request("POST", f"/{self.name}/_forcemerge", params={"max_num_segments": 1}, timeout=120)

    def search(self, query, admission, *, exact=False, limit=TOP_K):
        if not admission.admitted_count:
            return SearchResult(())
        body = _os_search_body(query, admission.allowed, admission.epoch, limit, admission.ef, exact)
        response = self._search(body)
        return SearchResult(tuple(int(hit["_source"]["point_id"]) for hit in response["hits"]["hits"]))

    def _search(self, body):
        response = self._request("POST", f"/{self.name}/_search", json=body)
        if response["timed_out"]:
            raise TimeoutError("OpenSearch query timed out")
        if response["_shards"]["failed"]:
            raise RuntimeError("OpenSearch query returned partial shard results")
        return response

    def profile(self, query, admission):
        body = _os_search_body(query, admission.allowed, admission.epoch, TOP_K, admission.ef, False)
        response = self._search({**body, "profile": True})
        hits = response["hits"]["hits"]
        if not hits:
            return parse_opensearch_plan(response, {})
        explanation = self._request(
            "POST", f"/{self.name}/_explain/{int(hits[0]['_id'])}", json={"query": body["query"]},
        )
        return parse_opensearch_plan(response, explanation)

    def drop(self):
        self._request("DELETE", f"/{self.name}")


class PgvectorPrefilter:
    name = "pgvector"

    def __init__(self, table, dsn):
        import psycopg

        self.table = _identifier(table)
        self.conn = psycopg.connect(
            dsn, autocommit=True, connect_timeout=TIMEOUT_SECONDS, cursor_factory=psycopg.ClientCursor,
        )

    def server_info(self):
        with self.conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("SELECT current_setting('server_version_num'), extversion "
                           "FROM pg_extension WHERE extname = 'vector'")
            row = cursor.fetchone()
        if row is None or int(row[0]) // 10000 != 17:
            raise RuntimeError("Bundle 48 requires PostgreSQL 17 with the vector extension installed")
        require_version(self.name, row[1])
        return {"version": row[1], "postgres_version_num": int(row[0])}

    def create(self, dim):
        if not 1 <= dim <= 2000:
            raise ValueError("pgvector vector HNSW supports dimensions 1 through 2000")
        self.dim = dim
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(f"CREATE TABLE {self.table} ("
                          f"point_id integer PRIMARY KEY, doc_id text NOT NULL, "
                          f"admitted_epoch integer NOT NULL, embedding vector({dim}) NOT NULL)")

    def ingest(self, vectors, epochs):
        with self.conn.cursor() as cursor:
            with cursor.copy(
                f"COPY {self.table} (point_id, doc_id, admitted_epoch, embedding) FROM STDIN"
            ) as copy:
                for i, (row, epoch) in enumerate(zip(vectors, epochs, strict=True)):
                    copy.write_row((i, str(i), int(epoch), json.dumps(row.tolist())))
        self.conn.execute("SET maintenance_work_mem = '512MB'")
        self.conn.execute(
            f"CREATE INDEX {self.table}_hnsw ON {self.table} USING hnsw (embedding vector_ip_ops) "
            "WITH (m = 64, ef_construction = 512)"
        )
        self.conn.execute(f"ANALYZE {self.table}")

    def _query(self, query, admission, *, exact, limit, explain=False):
        sql, params = pgvector_search_sql(
            self.table, PG_HNSW, query=query, allowed=admission.allowed,
            epoch=admission.epoch, limit=limit, exact=exact,
        )
        settings = [f"SET LOCAL statement_timeout = '{TIMEOUT_SECONDS}s'"]
        if exact:
            settings.extend(("SET LOCAL enable_indexscan = off", "SET LOCAL enable_indexonlyscan = off",
                             "SET LOCAL enable_bitmapscan = off"))
        else:
            # Discourage alternate filter scans and inspect the executed plan.
            settings.extend(("SET LOCAL hnsw.iterative_scan = 'relaxed_order'",
                             "SET LOCAL hnsw.max_scan_tuples = 200000",
                             "SET LOCAL hnsw.scan_mem_multiplier = 8",
                             "SET LOCAL hnsw.ef_search = %(ef_search)s",
                             "SET LOCAL enable_seqscan = off", "SET LOCAL enable_bitmapscan = off"))
            params["ef_search"] = admission.ef
        statement = ["SHOW hnsw.iterative_scan", "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql] if explain else [sql]
        # ClientCursor binds values and sends one simple-query message. PostgreSQL
        # executes the statements in one implicit transaction, so settings vanish
        # at message completion and the exact arm sees its own settings.
        with self.conn.cursor() as cursor:
            cursor.execute(";\n".join((*settings, *statement)), params)
            for _ in settings:
                if not cursor.nextset():
                    raise RuntimeError("pgvector query result set was missing")
            if explain:
                iterative_scan = cursor.fetchone()[0]
                if not cursor.nextset():
                    raise RuntimeError("pgvector EXPLAIN result set was missing")
                return parse_pgvector_plan(
                    cursor.fetchone()[0], hnsw_index=f"{self.table}_hnsw", iterative_scan=iterative_scan,
                    expected_hits=min(limit, admission.admitted_count),
                )
            return SearchResult(tuple(int(row[1]) for row in cursor.fetchall()))

    def search(self, query, admission, *, exact=False, limit=TOP_K):
        if not admission.admitted_count:
            return SearchResult(())
        return self._query(query, admission, exact=exact, limit=limit)

    def profile(self, query, admission):
        return self._query(query, admission, exact=False, limit=TOP_K, explain=True)

    def drop(self):
        self.conn.execute(f"DROP TABLE {self.table}")

    def close(self):
        self.conn.close()
