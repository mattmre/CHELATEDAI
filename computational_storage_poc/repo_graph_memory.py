from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Research-stage / POC module. No production code path in this repo consumes it.
EXPERIMENTAL = True

EMBEDDING_DIM = 256
SUPPORTED_EXTENSIONS = {".py", ".md", ".txt", ".js", ".ts"}
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IDENTIFIER_SUBTOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
FLOAT32_EMBEDDING_DTYPE = "float32"
INT8_EMBEDDING_DTYPE = "int8"
INT8_QUERY_CHUNK_ROWS = 64


@dataclass(frozen=True)
class CodeNode:
    node_id: int
    path: str
    kind: str
    language: str
    text: str
    tokens: list[str]
    symbol_name: str | None = None
    parent_id: int | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class GraphEdge:
    src: int
    dst: int
    edge_type: str


@dataclass(frozen=True)
class QueryResult:
    node_id: int
    path: str
    kind: str
    score: float
    embedding_score: float
    lexical_score: float
    graph_score: float
    path_score: float


def _recall_score(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _precision_score(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(candidate_tokens)


def _stable_hash(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        raw_token = match.group(0)
        for chunk in raw_token.split("_"):
            if not chunk:
                continue
            subparts = IDENTIFIER_SUBTOKEN_PATTERN.findall(chunk)
            if subparts:
                tokens.extend(part.lower() for part in subparts)
            else:
                tokens.append(chunk.lower())
    return tokens


def _embed_tokens(tokens: list[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        vector[_stable_hash(token) % dim] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def _quantize_embedding_matrix(embeddings: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(embeddings)))
    if max_abs == 0.0:
        return np.zeros_like(embeddings, dtype=np.int8), 1.0

    scale = max_abs / 127.0
    quantized = np.clip(np.rint(embeddings / scale), -127, 127).astype(np.int8)
    return quantized, scale


def _quantize_query_vector(vector: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(vector)))
    if max_abs == 0.0:
        return np.zeros_like(vector, dtype=np.int8), 1.0

    scale = max_abs / 127.0
    quantized = np.clip(np.rint(vector / scale), -127, 127).astype(np.int8)
    return quantized, scale


def _module_name_for_path(relative_path: str) -> str | None:
    path = Path(relative_path)
    if path.suffix != ".py":
        return None
    module = ".".join(path.with_suffix("").parts)
    if module.endswith("__init__"):
        module = module[: -len(".__init__")]
    return module


def _extract_python_symbol_nodes(file_node: CodeNode) -> tuple[list[CodeNode], list[str]]:
    try:
        tree = ast.parse(file_node.text)
    except SyntaxError:
        return [], []

    symbol_nodes: list[CodeNode] = []
    imported_modules: list[str] = []
    next_node_id = file_node.node_id + 1
    lines = file_node.text.splitlines()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = int(getattr(node, "lineno", 1))
            end_line = int(getattr(node, "end_lineno", start_line))
            snippet = "\n".join(lines[start_line - 1 : end_line])
            path_tokens = _tokenize_text(file_node.path.replace(os.sep, " "))
            symbol_tokens = _tokenize_text(f"{node.name} {snippet}") + path_tokens
            symbol_nodes.append(
                CodeNode(
                    node_id=next_node_id,
                    path=file_node.path,
                    kind="symbol",
                    language=file_node.language,
                    text=snippet,
                    tokens=symbol_tokens,
                    symbol_name=node.name,
                    parent_id=file_node.node_id,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            next_node_id += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    return symbol_nodes, imported_modules


def ingest_repo_graph_memory(
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    embedding_dim: int = EMBEDDING_DIM,
    exclude_substrings: tuple[str, ...] = (),
    embedding_storage_dtype: str = FLOAT32_EMBEDDING_DTYPE,
) -> dict[str, int | float | str]:
    source_root = Path(repo_root).resolve()
    target_root = Path(output_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    file_nodes: list[CodeNode] = []
    symbol_nodes: list[CodeNode] = []
    edges: list[GraphEdge] = []
    file_node_by_path: dict[str, CodeNode] = {}
    imported_modules_by_node: dict[int, list[str]] = {}
    next_node_id = 0

    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = str(path.relative_to(source_root)).replace("\\", "/")
        if exclude_substrings and any(fragment in relative_path for fragment in exclude_substrings):
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        language = "python" if path.suffix.lower() == ".py" else path.suffix.lower().lstrip(".")
        path_tokens = _tokenize_text(relative_path.replace("/", " "))
        node_tokens = _tokenize_text(text) + path_tokens
        file_node = CodeNode(
            node_id=next_node_id,
            path=relative_path,
            kind="file",
            language=language,
            text=text,
            tokens=node_tokens,
        )
        next_node_id += 1
        file_nodes.append(file_node)
        file_node_by_path[relative_path] = file_node

        if language == "python":
            extracted_symbols, imported_modules = _extract_python_symbol_nodes(file_node)
            imported_modules_by_node[file_node.node_id] = imported_modules
            for symbol_node in extracted_symbols:
                symbol_nodes.append(symbol_node)
                edges.append(GraphEdge(src=file_node.node_id, dst=symbol_node.node_id, edge_type="contains"))
                edges.append(GraphEdge(src=symbol_node.node_id, dst=file_node.node_id, edge_type="contained_by"))
            if extracted_symbols:
                next_node_id = max(next_node_id, extracted_symbols[-1].node_id + 1)

    module_to_node_id = {
        module_name: node.node_id
        for path, node in file_node_by_path.items()
        for module_name in [_module_name_for_path(path)]
        if module_name is not None
    }

    for file_node in file_nodes:
        for module_name in imported_modules_by_node.get(file_node.node_id, []):
            target_id = module_to_node_id.get(module_name)
            if target_id is None:
                continue
            edges.append(GraphEdge(src=file_node.node_id, dst=target_id, edge_type="imports"))
            edges.append(GraphEdge(src=target_id, dst=file_node.node_id, edge_type="imported_by"))

    all_nodes = file_nodes + symbol_nodes
    embeddings = np.vstack([_embed_tokens(node.tokens, dim=embedding_dim) for node in all_nodes])
    if embedding_storage_dtype == FLOAT32_EMBEDDING_DTYPE:
        stored_embeddings = embeddings.astype(np.float32, copy=False)
        embedding_scale = 1.0
    elif embedding_storage_dtype == INT8_EMBEDDING_DTYPE:
        stored_embeddings, embedding_scale = _quantize_embedding_matrix(embeddings)
    else:
        raise ValueError(f"Unsupported embedding storage dtype: {embedding_storage_dtype}")

    manifest = {
        "version": 1,
        "repo_root": str(source_root),
        "embedding_dim": embedding_dim,
        "embedding_storage_dtype": embedding_storage_dtype,
        "embedding_scale": embedding_scale,
        "node_count": len(all_nodes),
        "edge_count": len(edges),
        "created_at_epoch": time.time(),
    }

    with (target_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with (target_root / "nodes.jsonl").open("w", encoding="utf-8") as handle:
        for node in all_nodes:
            handle.write(json.dumps(asdict(node), sort_keys=True))
            handle.write("\n")
    with (target_root / "edges.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(edge) for edge in edges], handle, indent=2, sort_keys=True)
    np.save(target_root / "embeddings.npy", stored_embeddings)

    return {
        "repo_root": str(source_root),
        "memory_dir": str(target_root),
        "node_count": len(all_nodes),
        "edge_count": len(edges),
        "embedding_dim": embedding_dim,
        "embedding_storage_dtype": embedding_storage_dtype,
    }


class DiskBackedRepoGraphMemory:
    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        with (self.memory_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        self.nodes = self._load_nodes(self.memory_dir / "nodes.jsonl")
        self.node_by_id = {node.node_id: node for node in self.nodes}
        self.edges = self._load_edges(self.memory_dir / "edges.json")
        self.adjacency = self._build_adjacency(self.edges)
        self.embeddings = np.load(self.memory_dir / "embeddings.npy", mmap_mode="r")
        self.embedding_storage_dtype = str(self.manifest.get("embedding_storage_dtype", FLOAT32_EMBEDDING_DTYPE))
        self.embedding_scale = float(self.manifest.get("embedding_scale", 1.0))

    def close(self) -> None:
        embeddings = getattr(self, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "_mmap") and embeddings._mmap is not None:
            embeddings._mmap.close()
        self.embeddings = None

    def __enter__(self) -> "DiskBackedRepoGraphMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _load_nodes(path: Path) -> list[CodeNode]:
        nodes: list[CodeNode] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                nodes.append(CodeNode(**payload))
        return nodes

    @staticmethod
    def _load_edges(path: Path) -> list[GraphEdge]:
        with path.open("r", encoding="utf-8") as handle:
            return [GraphEdge(**payload) for payload in json.load(handle)]

    @staticmethod
    def _build_adjacency(edges: list[GraphEdge]) -> dict[int, list[GraphEdge]]:
        adjacency: dict[int, list[GraphEdge]] = {}
        for edge in edges:
            adjacency.setdefault(edge.src, []).append(edge)
        return adjacency

    @property
    def mapped_bytes(self) -> int:
        return int(self.embeddings.nbytes)

    def query(self, query_text: str, *, top_k: int = 5, rerank_depth: int = 12) -> list[QueryResult]:
        query_tokens = _tokenize_text(query_text)
        query_embedding = _embed_tokens(query_tokens, dim=int(self.manifest["embedding_dim"]))
        if self.embedding_storage_dtype == FLOAT32_EMBEDDING_DTYPE:
            embedding_scores = self.embeddings @ query_embedding
        elif self.embedding_storage_dtype == INT8_EMBEDDING_DTYPE:
            quantized_query, query_scale = _quantize_query_vector(query_embedding)
            query_vector = quantized_query.astype(np.int16, copy=False)
            embedding_scores = np.empty(self.embeddings.shape[0], dtype=np.float32)
            for start in range(0, self.embeddings.shape[0], INT8_QUERY_CHUNK_ROWS):
                end = min(start + INT8_QUERY_CHUNK_ROWS, self.embeddings.shape[0])
                score_chunk = self.embeddings[start:end].astype(np.int16) @ query_vector
                embedding_scores[start:end] = score_chunk.astype(np.float32) * (self.embedding_scale * query_scale)
        else:
            raise ValueError(f"Unsupported embedding storage dtype: {self.embedding_storage_dtype}")

        initial_order = np.argsort(-embedding_scores)[: min(rerank_depth, len(self.nodes))]
        query_token_set = set(query_tokens)
        results: list[QueryResult] = []

        for node_index in initial_order:
            node = self.nodes[int(node_index)]
            node_token_set = set(node.tokens)
            lexical_score = _recall_score(query_token_set, node_token_set)
            path_token_set = set(_tokenize_text(node.path))
            path_score = _recall_score(query_token_set, path_token_set)
            path_precision_score = _precision_score(query_token_set, path_token_set)
            basename_token_set = set(_tokenize_text(Path(node.path).stem))
            basename_precision_score = _precision_score(query_token_set, basename_token_set)
            unmatched_path_penalty = max(0.0, 1.0 - path_precision_score)

            neighbor_tokens: set[str] = set()
            for edge in self.adjacency.get(node.node_id, []):
                neighbor = self.node_by_id.get(edge.dst)
                if neighbor is not None:
                    neighbor_tokens.update(neighbor.tokens)
            graph_score = _recall_score(query_token_set, neighbor_tokens)

            embedding_score = float(embedding_scores[int(node_index)])
            kind_bonus = 0.05 if node.kind == "file" else 0.0
            final_score = (
                (0.38 * embedding_score)
                + (0.18 * lexical_score)
                + (0.08 * graph_score)
                + (0.14 * path_score)
                + (0.10 * path_precision_score)
                + (0.12 * basename_precision_score)
                + kind_bonus
                - (0.05 * unmatched_path_penalty)
            )
            results.append(
                QueryResult(
                    node_id=node.node_id,
                    path=node.path,
                    kind=node.kind,
                    score=final_score,
                    embedding_score=embedding_score,
                    lexical_score=lexical_score,
                    graph_score=graph_score,
                    path_score=path_score,
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        unique_results: list[QueryResult] = []
        seen_paths: set[str] = set()
        for result in results:
            if result.path in seen_paths:
                continue
            unique_results.append(result)
            seen_paths.add(result.path)
            if len(unique_results) >= top_k:
                break
        return unique_results
