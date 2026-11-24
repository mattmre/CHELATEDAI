"""Wave-11 D7 live Graphiti / Zep arena primitives (graphiti-core 0.31.2).

Single-database Neo4j Community shim, metered vLLM chat-completions client,
GPU BGE-M3 embedder, GPU BGE reranker, active-edge reader formatting,
Delta=0s versus settled choreography, and Suite A / Suite B score tables.

Live GPU, Docker, and SSH execution stay in ``scripts/run_wave11_d7_live_zep_graphiti_arena.py``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ValidationError

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import EmptyResponseError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://127.0.0.1:8891/v1"
VLLM_MODEL = "nemotron-lightning"
COMMUNITY_DATABASE = "neo4j"
EMBEDDING_DIM = 1024
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_COROUTINES = 6
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 42

ARM_CHELATED = "Chelated_v4_Stratified_WFS"
ARM_GRAPHITI = "Live_Graphiti_Zep_v0_31_2"
ARM_MEM0 = "Live_Mem0_v2_2_1_InferTrue"
ARENA_ARMS = (ARM_CHELATED, ARM_GRAPHITI, ARM_MEM0)

GROUP_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_RETRACTION_MARKERS = (
    "official_supersede",
    "official_debunk",
    "v3_final",
    "d01_party_b",
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
_SCHEMA_NOTE = "\n\nRespond with a JSON object in the following format:\n\n"

NODE_COUNT_CYPHER = "MATCH (n {group_id: $gid}) RETURN count(n) AS n_cnt"
EDGE_COUNT_CYPHER = "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() RETURN count(r) AS r_cnt"


def case_group_id(idx: int, *, prefix: str = "d7_case") -> str:
    """Return a Community-safe property group id. Never a database name."""
    if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx > 99:
        raise ValueError(f"case index must be an int in 0..99, got {idx!r}")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,24}", prefix):
        raise ValueError(f"unsafe group-id prefix {prefix!r}")
    gid = f"{prefix}_{idx:02d}"
    if GROUP_ID_RE.fullmatch(gid) is None or gid == COMMUNITY_DATABASE:
        raise ValueError(f"refusing group_id {gid!r}")
    return gid


def require_group_id(gid: str) -> str:
    if not isinstance(gid, str) or GROUP_ID_RE.fullmatch(gid) is None or gid in {"", COMMUNITY_DATABASE}:
        raise ValueError(f"refusing empty or database-shaped group_id {gid!r}")
    return gid


def http_status(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def repair_model_json(raw: str) -> Tuple[str, int]:
    """Strip ``<think>`` blocks and markdown fences. Return ``(json_text, repair_ops)``."""
    text = raw or ""
    repairs = 0
    if _THINK_BLOCK_RE.search(text) or _THINK_OPEN_RE.search(text):
        repairs += 1
        text = _THINK_BLOCK_RE.sub("", text)
        text = _THINK_OPEN_RE.sub("", text)
    stripped = text.strip()
    if "```" in stripped:
        repairs += 1
        stripped = OpenAIGenericClient._strip_code_fences(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        if candidate != stripped:
            repairs += 1
            stripped = candidate
    return stripped.strip(), repairs


class MeteredVllmOpenAIGenericClient(OpenAIGenericClient):
    """OpenAI-compatible chat client for ``nemotron-lightning`` on ``:8891``.

    Every ``chat.completions.create`` call forces ``temperature=0.0`` and
    ``extra_body.chat_template_kwargs.enable_thinking=False``. ``json_schema``
    is probed first. HTTP 400 or a non-object body switches the client to
    ``json_object`` and retries once. Repaired JSON is schema-validated before
    return; a ``ValidationError`` retries once via ``_build_response_format``.
    ``json_object`` retries include ``model_json_schema()`` in the prompt.
    ``finish_reason == "length"`` increments ``length_truncation_count``.
    Missing ``response.usage`` raises ``RuntimeError``.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cache: bool = False,
        client: Any = None,
        max_tokens: int = 4096,
        structured_output_mode: str = "json_schema",
    ) -> None:
        if config is None:
            config = LLMConfig(api_key="local-vllm", model=VLLM_MODEL, base_url=VLLM_BASE_URL, temperature=0.0)
        base = str(config.base_url or "").rstrip("/")
        if base != VLLM_BASE_URL:
            raise ValueError(f"Metered vLLM client refuses base_url {base!r}; expected {VLLM_BASE_URL}")
        super().__init__(
            config=config,
            cache=cache,
            client=client,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,  # type: ignore[arg-type]
        )
        self.temperature = 0.0
        self.config.temperature = 0.0
        self.schema_fallback_count = 0
        self.json_repair_count = 0
        self.validation_retry_count = 0
        self.length_truncation_count = 0
        self.write_prompt_tokens = 0
        self.write_completion_tokens = 0
        self.read_prompt_tokens = 0
        self.read_completion_tokens = 0
        self.llm_call_count = 0
        self.phase = "write"
        self._schema_fallback_locked = False
        self._install_meter()

    def reset_counters(self, *, phase: str) -> None:
        if phase not in {"write", "read"}:
            raise ValueError(f"phase must be 'write' or 'read', got {phase!r}")
        self.phase = phase
        self.llm_call_count = 0
        # schema_fallback_count, json_repair_count, validation_retry_count, and
        # length_truncation_count survive phase resets.
        if phase == "write":
            self.write_prompt_tokens = 0
            self.write_completion_tokens = 0
        else:
            self.read_prompt_tokens = 0
            self.read_completion_tokens = 0

    def token_snapshot(self) -> Dict[str, int]:
        return {
            "phase": self.phase,  # type: ignore[dict-item]
            "write_prompt_tokens": self.write_prompt_tokens,
            "write_completion_tokens": self.write_completion_tokens,
            "read_prompt_tokens": self.read_prompt_tokens,
            "read_completion_tokens": self.read_completion_tokens,
            "llm_call_count": self.llm_call_count,
            "schema_fallback_count": self.schema_fallback_count,
            "json_repair_count": self.json_repair_count,
            "validation_retry_count": self.validation_retry_count,
            "length_truncation_count": self.length_truncation_count,
        }

    def _install_meter(self) -> None:
        original = self.client.chat.completions.create

        async def metered_create(*args: Any, **kwargs: Any) -> Any:
            kwargs["temperature"] = 0.0
            extra = dict(kwargs.get("extra_body") or {})
            template_kwargs = dict(extra.get("chat_template_kwargs") or {})
            template_kwargs["enable_thinking"] = False
            extra["chat_template_kwargs"] = template_kwargs
            kwargs["extra_body"] = extra
            if not kwargs.get("model"):
                kwargs["model"] = self.model or VLLM_MODEL
            response = await original(*args, **kwargs)
            self._account_usage(response)
            return response

        self.client.chat.completions.create = metered_create

    def _account_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            raise RuntimeError("vLLM chat completion is missing response.usage")
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is None or completion is None:
            raise RuntimeError("vLLM response.usage is missing prompt_tokens or completion_tokens")
        self.llm_call_count += 1
        if self.phase == "write":
            self.write_prompt_tokens += int(prompt)
            self.write_completion_tokens += int(completion)
        elif self.phase == "read":
            self.read_prompt_tokens += int(prompt)
            self.read_completion_tokens += int(completion)
        else:
            raise RuntimeError(f"unknown meter phase {self.phase!r}")

    @staticmethod
    def _is_empty_object_model(response_model: Optional[type[BaseModel]]) -> bool:
        if response_model is None:
            return False
        schema = response_model.model_json_schema()
        props = schema.get("properties")
        return isinstance(props, dict) and len(props) == 0 and not schema.get("$defs")

    def _build_response_format(self, response_model: Optional[type[BaseModel]]) -> Dict[str, Any]:
        if response_model is None or self.structured_output_mode == "json_object":
            return {"type": "json_object"}
        if self._is_empty_object_model(response_model):
            return {"type": "json_object"}
        return super()._build_response_format(response_model)

    def _append_schema(self, messages: Sequence[Message], response_model: Optional[type[BaseModel]]) -> None:
        if response_model is None or not messages:
            return
        if self._is_empty_object_model(response_model):
            empty_note = "\n\nRespond with an empty JSON object: {}"
            if empty_note.strip() not in (messages[-1].content or ""):
                messages[-1].content = f"{messages[-1].content or ''}{empty_note}"
            return
        schema = json.dumps(response_model.model_json_schema())
        inst_note = (
            "\n\nRespond with a valid JSON instance (never output the JSON Schema itself; "
            f"do not include '$defs' or 'properties' keys) matching:\n\n{schema}"
        )
        if inst_note.strip() in (messages[-1].content or "") or schema in (messages[-1].content or ""):
            return
        messages[-1].content = f"{messages[-1].content or ''}{inst_note}"

    async def _switch_to_json_object(
        self,
        messages: List[Message],
        response_model: Optional[type[BaseModel]],
        max_tokens: int,
        model_size: Any,
        model: Optional[str],
    ) -> Dict[str, Any]:
        self.schema_fallback_count += 1
        self.structured_output_mode = "json_object"
        self._schema_fallback_locked = True
        logger.warning(
            "graphiti vLLM json_schema fallback -> json_object (schema_fallback_count=%s)",
            self.schema_fallback_count,
        )
        try:
            return await self._generate_response(
                messages,
                response_model,
                max_tokens,
                model_size,
                model=model,
            )
        finally:
            self._schema_fallback_locked = False

    async def probe_json_schema(self) -> str:
        """One structured probe. Returns the mode that will be used afterwards."""

        class _Probe(BaseModel):
            ok: bool

        before = self.structured_output_mode
        if before != "json_schema":
            return str(before)
        messages = [
            Message(role="system", content="Reply with one JSON object and no prose."),
            Message(role="user", content='Return {"ok": true}'),
        ]
        await self._generate_response(messages, _Probe, max_tokens=64)
        return str(self.structured_output_mode)

    def _record_length_truncation(self, response: Any) -> None:
        finish = getattr(response.choices[0], "finish_reason", None)
        if finish == "length":
            self.length_truncation_count += 1

    @staticmethod
    def _with_json_object_schema(content: str, response_model: type[BaseModel]) -> str:
        schema_text = json.dumps(response_model.model_json_schema())
        if schema_text in content:
            return content
        return f"{content}\n\n{schema_text}"

    @staticmethod
    def _copied_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [{"role": item["role"], "content": item["content"]} for item in messages]

    async def _retry_after_validation_error(
        self,
        openai_messages: List[Dict[str, str]],
        response_model: type[BaseModel],
        *,
        max_tokens: int,
        model: Optional[str],
    ) -> Dict[str, Any]:
        self.validation_retry_count += 1
        retry_format = self._build_response_format(response_model)
        anti_echo = (
            "\n\nIMPORTANT: Output ONLY a concrete JSON data object satisfying the fields, "
            "NOT the JSON Schema definition (never output '$defs' or 'properties')."
        )
        content = f"{openai_messages[-1]['content']}{anti_echo}"
        if retry_format.get("type") == "json_object":
            content = self._with_json_object_schema(content, response_model)
        retry_messages = self._copied_messages(openai_messages)
        retry_messages[-1] = {"role": retry_messages[-1]["role"], "content": content}
        request_model = model or self.model or VLLM_MODEL
        try:
            retry_resp = await self.client.chat.completions.create(
                model=request_model,
                messages=self._copied_messages(retry_messages),
                temperature=0.0,
                max_tokens=max_tokens,
                response_format=retry_format,
            )
        except Exception as exc:
            if (
                http_status(exc) == 400
                and retry_format.get("type") == "json_schema"
                and not self._schema_fallback_locked
            ):
                self.schema_fallback_count += 1
                self.structured_output_mode = "json_object"
                retry_format = {"type": "json_object"}
                content = self._with_json_object_schema(content, response_model)
                retry_messages[-1] = {"role": retry_messages[-1]["role"], "content": content}
                retry_resp = await self.client.chat.completions.create(
                    model=request_model,
                    messages=self._copied_messages(retry_messages),
                    temperature=0.0,
                    max_tokens=max_tokens,
                    response_format=retry_format,
                )
            else:
                raise
        self._record_length_truncation(retry_resp)
        retry_raw = retry_resp.choices[0].message.content or ""
        retry_repaired, retry_n = repair_model_json(str(retry_raw))
        if retry_n:
            self.json_repair_count += int(retry_n)
        parsed = json.loads(retry_repaired)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"vLLM retry JSON object required, got {type(parsed).__name__}")
        response_model.model_validate(parsed)
        return parsed

    async def _generate_response(
        self,
        messages: List[Message],
        response_model: Optional[type[BaseModel]] = None,
        max_tokens: int = 4096,
        model_size: Any = None,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        # ``model_size`` is part of the Graphiti signature. vLLM serves one model id,
        # and the name stays bound so the json_object retry can forward it.
        effective_max_tokens = min(max(int(max_tokens or 4096), 1), 4096)
        if response_model is not None and (
            self.structured_output_mode == "json_object" or self._is_empty_object_model(response_model)
        ):
            self._append_schema(messages, response_model)
        openai_messages: List[Dict[str, str]] = []
        for item in messages:
            item.content = self._clean_input(item.content)
            if item.role in {"user", "system"}:
                openai_messages.append({"role": item.role, "content": item.content})
        if not openai_messages:
            raise RuntimeError("refusing an empty chat completion")
        req_format = self._build_response_format(response_model)
        if req_format.get("type") == "json_object":
            joined_lower = " ".join(m["content"].lower() for m in openai_messages)
            if "json" not in joined_lower:
                openai_messages[-1] = {
                    "role": openai_messages[-1]["role"],
                    "content": f"{openai_messages[-1]['content']}\nRespond with a valid JSON object.",
                }
        try:
            response = await self.client.chat.completions.create(
                model=model or self.model or VLLM_MODEL,
                messages=openai_messages,
                temperature=0.0,
                max_tokens=effective_max_tokens,
                response_format=req_format,
            )
        except Exception as exc:
            if (
                http_status(exc) == 400
                and self.structured_output_mode == "json_schema"
                and not self._schema_fallback_locked
            ):
                return await self._switch_to_json_object(
                    messages, response_model, effective_max_tokens, model_size, model
                )
            raise
        self._record_length_truncation(response)
        content = response.choices[0].message.content or ""
        if not str(content).strip():
            raise EmptyResponseError("LLM returned an empty response")
        repaired, n_rep = repair_model_json(str(content))
        if n_rep:
            self.json_repair_count += int(n_rep)
        parse_error: Optional[BaseException] = None
        parsed: Any = None
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as exc:
            parse_error = exc
        object_ok = parse_error is None and isinstance(parsed, dict)
        saw_schema_problem = (not object_ok) or n_rep > 0
        needs_validation_retry = False
        if (
            saw_schema_problem
            and self.structured_output_mode == "json_schema"
            and not self._schema_fallback_locked
        ):
            if object_ok:
                self.schema_fallback_count += 1
                self.structured_output_mode = "json_object"
                if response_model is not None and not self._is_empty_object_model(response_model):
                    try:
                        response_model.model_validate(parsed)
                    except ValidationError:
                        logger.warning(
                            "graphiti vLLM repaired JSON failed validation; retrying as json_object "
                            "(schema_fallback_count=%s, json_repair_count=%s)",
                            self.schema_fallback_count,
                            self.json_repair_count,
                        )
                        needs_validation_retry = True
                    else:
                        logger.warning(
                            "graphiti vLLM kept repaired JSON and locked json_object "
                            "(schema_fallback_count=%s, json_repair_count=%s)",
                            self.schema_fallback_count,
                            self.json_repair_count,
                        )
                        return parsed
                else:
                    logger.warning(
                        "graphiti vLLM kept repaired JSON and locked json_object "
                        "(schema_fallback_count=%s, json_repair_count=%s)",
                        self.schema_fallback_count,
                        self.json_repair_count,
                    )
                    return parsed
            else:
                return await self._switch_to_json_object(
                    messages, response_model, effective_max_tokens, model_size, model
                )
        if not needs_validation_retry:
            if parse_error is not None:
                raise parse_error
            if not isinstance(parsed, dict):
                raise RuntimeError(f"vLLM JSON object required, got {type(parsed).__name__}")
            if response_model is not None and not self._is_empty_object_model(response_model):
                try:
                    response_model.model_validate(parsed)
                except ValidationError:
                    needs_validation_retry = True
        if needs_validation_retry and response_model is not None:
            parsed = await self._retry_after_validation_error(
                openai_messages,
                response_model,
                max_tokens=effective_max_tokens,
                model=model,
            )
        return parsed


def _as_unit_vectors(matrix: np.ndarray) -> List[List[float]]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise RuntimeError(f"encoder returned ndim={arr.ndim}")
    out: List[List[float]] = []
    for row in arr:
        if int(row.shape[0]) != EMBEDDING_DIM:
            raise RuntimeError(f"bge-m3 embedding length {int(row.shape[0])} != {EMBEDDING_DIM}")
        norm = float(np.linalg.norm(row))
        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("bge-m3 returned a zero or non-finite embedding")
        unit = row / norm
        if int(unit.shape[0]) != EMBEDDING_DIM:
            raise RuntimeError(f"normalized embedding length {int(unit.shape[0])} != {EMBEDDING_DIM}")
        out.append([float(value) for value in unit.tolist()])
    return out


class LocalGpuBgeM3Embedder(EmbedderClient):
    """Persistent GPU ``BAAI/bge-m3`` embedder. ``create`` and ``create_batch`` are both real."""

    def __init__(
        self,
        *,
        model_name: str = BGE_M3_MODEL,
        device: Optional[str] = None,
        encoder: Optional[Callable[[Sequence[str]], Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._encoder = encoder
        self._model: Any = None

    def _load(self) -> None:
        if self._encoder is not None or self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._model = SentenceTransformer(self.model_name, device=device)

    def encode_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("refusing to embed an empty batch")
        cleaned = [str(text) for text in texts]
        self._load()
        if self._encoder is not None:
            raw = self._encoder(cleaned)
        else:
            raw = self._model.encode(
                cleaned,
                batch_size=16,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        vectors = _as_unit_vectors(np.asarray(raw))
        if len(vectors) != len(cleaned):
            raise RuntimeError(f"encoder returned {len(vectors)} vectors for {len(cleaned)} texts")
        return vectors

    async def create(self, input_data: Any) -> List[float]:
        # Graphiti calls ``create(input_data=[text])`` and keeps only one vector.
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, (list, tuple)) and len(input_data) == 1 and isinstance(input_data[0], str):
            texts = [str(input_data[0])]
        else:
            raise TypeError(
                "LocalGpuBgeM3Embedder.create expects one string (or a one-item list). Use create_batch for many."
            )
        return (await asyncio.to_thread(self.encode_texts, texts))[0]

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        if not isinstance(input_data_list, list) or not input_data_list:
            raise ValueError("create_batch expects a non-empty list of strings")
        if not all(isinstance(item, str) for item in input_data_list):
            raise TypeError("create_batch expects list[str]")
        return await asyncio.to_thread(self.encode_texts, input_data_list)


class LocalGpuBgeReranker(CrossEncoderClient):
    """Persistent GPU ``BAAI/bge-reranker-v2-m3`` cross-encoder."""

    def __init__(
        self,
        *,
        model_name: str = BGE_RERANKER_MODEL,
        device: Optional[str] = None,
        predict_fn: Optional[Callable[[Sequence[Sequence[str]]], Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._predict_fn = predict_fn
        self._model: Any = None

    def _load(self) -> None:
        if self._predict_fn is not None or self._model is not None:
            return
        import torch
        from sentence_transformers import CrossEncoder

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._model = CrossEncoder(self.model_name, device=device)

    def predict_pairs(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        self._load()
        if self._predict_fn is not None:
            raw = self._predict_fn(pairs)
        else:
            raw = self._model.predict(list(pairs), batch_size=16, show_progress_bar=False)
        scores = [float(value) for value in np.asarray(raw, dtype=np.float64).reshape(-1).tolist()]
        if len(scores) != len(pairs):
            raise RuntimeError(f"reranker returned {len(scores)} scores for {len(pairs)} pairs")
        if not all(np.isfinite(score) for score in scores):
            raise RuntimeError("reranker returned a non-finite score")
        return scores

    async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        if not passages:
            return []
        pairs = [[query, passage] for passage in passages]
        scores = await asyncio.to_thread(self.predict_pairs, pairs)
        order = sorted(range(len(passages)), key=lambda index: (-scores[index], index))
        return [(passages[index], float(scores[index])) for index in order]


class SingleDbCommunityNeo4jDriver(Neo4jDriver):
    """Pin Neo4j 5.26 Community to the single ``neo4j`` database.

    ``clone(database=...)`` and ``with_database`` keep ``_database == "neo4j"``.
    Case isolation is the node/edge property ``group_id``, not ``CREATE DATABASE``.
    """

    def __init__(self, uri: str, user: Optional[str], password: Optional[str], database: str = COMMUNITY_DATABASE) -> None:
        if database != COMMUNITY_DATABASE:
            logger.warning(
                "Ignoring Neo4j database=%s; Community edition stays on %s",
                database,
                COMMUNITY_DATABASE,
            )
        super().__init__(uri, user, password, database=COMMUNITY_DATABASE)
        if self._database != COMMUNITY_DATABASE:
            raise RuntimeError(f"Neo4j driver database drifted to {self._database!r}")

    def clone(self, database: str) -> "SingleDbCommunityNeo4jDriver":
        if database != COMMUNITY_DATABASE:
            logger.warning(
                "Refusing clone(database=%s); pinning %s",
                database,
                COMMUNITY_DATABASE,
            )
        cloned = copy.copy(self)
        cloned._database = COMMUNITY_DATABASE
        return cloned

    def with_database(self, database: str) -> "SingleDbCommunityNeo4jDriver":
        return self.clone(database)

    def session(self, database: Optional[str] = None) -> Any:
        if database not in (None, COMMUNITY_DATABASE):
            logger.warning("Forcing session database from %s to %s", database, COMMUNITY_DATABASE)
        self._database = COMMUNITY_DATABASE
        return super().session(COMMUNITY_DATABASE)

    async def execute_query(self, cypher_query_: Any, **kwargs: Any) -> Any:
        requested = kwargs.get("database_", self._database)
        if requested not in (None, COMMUNITY_DATABASE):
            logger.warning("Forcing database_ from %s to %s", requested, COMMUNITY_DATABASE)
        kwargs["database_"] = COMMUNITY_DATABASE
        self._database = COMMUNITY_DATABASE
        return await super().execute_query(cypher_query_, **kwargs)


def pin_graphiti_request_scope(graphiti: Any) -> None:
    """Stop ``_resolve_request_scope`` from leaving database ``neo4j``."""
    original = graphiti._resolve_request_scope

    def _pinned(group_id: Optional[str]) -> Tuple[str, Any, Any]:
        gid, driver, clients = original(group_id)
        if gid in (None, ""):
            raise RuntimeError("group_id resolved empty; refusing to merge every case")
        require_group_id(str(gid))
        pinned = getattr(driver, "_database", None)
        if pinned != COMMUNITY_DATABASE:
            if not hasattr(driver, "clone"):
                raise RuntimeError(f"driver database {pinned!r} is not {COMMUNITY_DATABASE}")
            driver = driver.clone(COMMUNITY_DATABASE)
            clients = clients.model_copy(update={"driver": driver})
        if getattr(driver, "_database", None) != COMMUNITY_DATABASE:
            raise RuntimeError("single-db pin failed")
        return str(gid), driver, clients

    graphiti._resolve_request_scope = _pinned


def _cypher_scalar(result: Any, key: str) -> int:
    records = getattr(result, "records", None)
    if records is None and isinstance(result, tuple):
        records = result[0]
    if not records:
        raise RuntimeError(f"Cypher returned no records for {key}")
    row = records[0]
    if hasattr(row, "keys") and callable(row.keys):
        row_keys = tuple(row.keys())
        if key not in row_keys:
            raise RuntimeError(f"Cypher row missing {key}: {row!r}")
        return int(row[key])
    if isinstance(row, Mapping):
        if key not in row:
            raise RuntimeError(f"Cypher row missing {key}: {row!r}")
        return int(row[key])
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {key} from {row!r}") from exc


async def verify_case_graph_populated(driver: Any, gid: str, *, settled_gate: bool = True) -> Dict[str, Any]:
    """Count nodes and ``RELATES_TO`` edges for one ``group_id``.

    A settled case with ``r_cnt == 0`` is ``ARM_FAILURE``. Delta=0s passes
    ``settled_gate=False`` and may observe a zero count while the retraction
    write is still in flight. Cypher errors propagate.
    """
    group = require_group_id(gid)
    node_result = await driver.execute_query(NODE_COUNT_CYPHER, params={"gid": group})
    edge_result = await driver.execute_query(EDGE_COUNT_CYPHER, params={"gid": group})
    n_cnt = _cypher_scalar(node_result, "n_cnt")
    r_cnt = _cypher_scalar(edge_result, "r_cnt")
    if n_cnt < 0 or r_cnt < 0:
        raise RuntimeError(f"negative graph counts n={n_cnt} r={r_cnt}")
    if not settled_gate:
        status = "DELTA0_OBSERVED"
        reason = "in_flight"
    elif n_cnt > 0 and r_cnt > 0:
        status = "OK"
        reason = "populated"
    elif n_cnt > 0 and r_cnt == 0:
        status = "ARM_FAILURE"
        reason = "episode_only"
    else:
        status = "ARM_FAILURE"
        reason = "zero_edges" if r_cnt == 0 else "nodes_missing"
    return {
        "group_id": group,
        "database": COMMUNITY_DATABASE,
        "n_cnt": n_cnt,
        "r_cnt": r_cnt,
        "status": status,
        "reason": reason,
    }


def active_edge_search_filter() -> SearchFilters:
    return SearchFilters(invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]])


def edge_hybrid_cross_encoder_config(num_results: int) -> Any:
    if num_results < 1:
        raise ValueError("num_results must be positive")
    config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    config.limit = int(num_results)
    return config


def format_fact_line(edge: Any) -> str:
    valid_at = getattr(edge, "valid_at", None)
    if hasattr(valid_at, "isoformat"):
        shown = valid_at.isoformat()
    elif valid_at is None:
        shown = ""
    else:
        shown = str(valid_at)
    fact = str(getattr(edge, "fact", "") or "").replace("\n", " ").replace("\r", " ").strip()
    return f"[fact | valid_at={shown}] {fact}"


def filter_active_edges(edges: Iterable[Any], gid: str) -> List[Any]:
    group = require_group_id(gid)
    kept: List[Any] = []
    for edge in edges:
        edge_gid = getattr(edge, "group_id", None)
        if edge_gid != group:
            raise RuntimeError(f"search leaked group_id {edge_gid!r} != {group!r}")
        if getattr(edge, "invalid_at", None) is not None:
            continue
        kept.append(edge)
    return kept


async def format_active_facts(graphiti: Any, query: str, gid: str, *, num_results: int = 8) -> str:
    """Current facts only: edge hybrid search, ``invalid_at is None``, one group."""
    group = require_group_id(gid)
    if not str(query or "").strip():
        raise ValueError("refusing an empty search query")
    search_filter = active_edge_search_filter()
    config = edge_hybrid_cross_encoder_config(num_results)
    if not hasattr(graphiti, "search_"):
        raise RuntimeError("Graphiti instance has no search_; refusing COMBINED episode search")
    result = await graphiti.search_(
        query,
        config=config,
        group_ids=[group],
        search_filter=search_filter,
    )
    edges = list(getattr(result, "edges", None) or [])
    active = filter_active_edges(edges, group)
    if not active:
        return "(No active facts.)"
    return "\n".join(format_fact_line(edge) for edge in active[:num_results])


def check_stale_resurrection(context_text: str, stale_values: Sequence[str]) -> bool:
    """True when a retracted value appears as a substring. Separate from judge correctness."""
    low = (context_text or "").lower()
    for value in stale_values:
        cleaned = str(value or "").strip().lower()
        if cleaned and len(cleaned) >= 3 and cleaned in low:
            return True
    return False


def split_stale_leakage(context_text: str, answer_text: str, stale_values: Sequence[str]) -> Dict[str, bool]:
    return {
        "stale_context": check_stale_resurrection(context_text, stale_values),
        "stale_answer": check_stale_resurrection(answer_text, stale_values),
    }


def select_retraction_document(docs: Sequence[Any]) -> Any:
    if not docs:
        raise ValueError("no documents")
    hits = [
        doc
        for doc in docs
        if any(marker in str(getattr(doc, "doc_id", "")) for marker in _RETRACTION_MARKERS)
    ]
    if not hits:
        raise ValueError("no retraction/supersession document identified")

    def _key(doc: Any) -> Tuple[int, int, str]:
        return (
            int(getattr(doc, "ingested_ts", 0)),
            int(getattr(doc, "published_ts", 0)),
            str(getattr(doc, "doc_id", "")),
        )

    return max(hits, key=_key)


def partition_retraction_documents(docs: Sequence[Any]) -> Tuple[List[Any], Any]:
    retraction = select_retraction_document(docs)
    retraction_id = str(getattr(retraction, "doc_id", ""))
    ordered = sorted(
        docs,
        key=lambda doc: (
            int(getattr(doc, "ingested_ts", 0)),
            int(getattr(doc, "published_ts", 0)),
            str(getattr(doc, "doc_id", "")),
        ),
    )
    prior = [doc for doc in ordered if str(getattr(doc, "doc_id", "")) != retraction_id]
    if len(prior) + 1 != len(list(docs)):
        raise RuntimeError(f"retraction partition lost documents for {retraction_id}")
    return prior, retraction


def reference_time_from_ts(ts: float) -> datetime:
    value = float(ts)
    if value > 10_000_000:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=int(value))


def knowledge_update_retraction_text(item: Mapping[str, Any]) -> str:
    answer_sids = set(item.get("answer_session_ids") or [])
    last = ""
    sessions = item.get("haystack_sessions") or []
    sids = item.get("haystack_session_ids") or []
    for sid, turns in zip(sids, sessions):
        if sid not in answer_sids:
            continue
        for turn in turns:
            if turn.get("has_answer") and str(turn.get("role") or "user").lower() == "user":
                last = str(turn.get("content") or "").strip()
    if not last:
        raise RuntimeError(f"knowledge-update item {item.get('question_id')!r} has no answer turn")
    return last


def select_retraction_message(messages: Sequence[Mapping[str, Any]], retraction_text: str) -> Mapping[str, Any]:
    needle = retraction_text.strip()
    matches = [
        message
        for message in messages
        if (not message.get("echo")) and str(message.get("content") or "").strip() == needle
    ]
    if not matches:
        raise RuntimeError("retraction text is not in the non-echo message list")
    return max(matches, key=lambda message: (float(message.get("ts") or 0.0), str(message.get("session_id") or "")))


def select_suite_b_items(records: Sequence[Mapping[str, Any]], *, n: int = 20) -> List[Mapping[str, Any]]:
    chosen: List[Mapping[str, Any]] = []
    ordered = sorted(records, key=lambda item: str(item.get("question_id") or ""))
    for item in ordered:
        if item.get("question_type") != "knowledge-update":
            continue
        if str(item.get("question_id") or "").endswith("_abs"):
            continue
        chosen.append(item)
        if len(chosen) == n:
            break
    if len(chosen) != n:
        raise RuntimeError(f"Suite B selection produced {len(chosen)} knowledge-update items, expected {n}")
    return chosen


async def run_delta0_then_settled(
    start_retraction: Callable[[], Awaitable[Any]],
    search_once: Callable[[], Awaitable[Any]],
) -> Dict[str, Any]:
    """Schedule the retraction, search immediately, then search again after ``await``.

    ``task_done_at_delta0`` is captured before the first await so a still-pending
    ``asyncio.Task`` is visible. The write task is awaited even when the first
    search fails, and its exception is not discarded.
    """
    started = time.perf_counter()
    task = asyncio.create_task(start_retraction())
    write_ack_ms = (time.perf_counter() - started) * 1000.0
    task_done_at_delta0 = bool(task.done())
    finished_ms: Dict[str, float] = {}

    def _on_done(done_task: asyncio.Task[Any]) -> None:
        finished_ms["ms"] = (time.perf_counter() - started) * 1000.0
        # Retrieve the exception later via ``await``. Touching it here would
        # mark the failure observed before the caller can raise it.
        del done_task

    task.add_done_callback(_on_done)
    delta0_error: Optional[BaseException] = None
    delta0: Any = None
    try:
        delta0 = await search_once()
    except Exception as exc:
        delta0_error = exc
    task_error: Optional[BaseException] = None
    try:
        retraction_value = await task
    except Exception as exc:
        task_error = exc
        retraction_value = None
    consolidation_bg_ms = float(finished_ms.get("ms", (time.perf_counter() - started) * 1000.0))
    if task_error is not None:
        raise task_error
    if delta0_error is not None:
        raise delta0_error
    read_started = time.perf_counter()
    settled = await search_once()
    settled_read_ms = (time.perf_counter() - read_started) * 1000.0
    return {
        "delta0": delta0,
        "settled": settled,
        "retraction_value": retraction_value,
        "task_done_at_delta0": task_done_at_delta0,
        "task_done_after_settle": bool(task.done()),
        "write_ack_ms": write_ack_ms,
        "consolidation_bg_ms": consolidation_bg_ms,
        "settled_read_ms": settled_read_ms,
        "synchronous_ek5_fence": False,
    }


def synchronous_fence_horizons(context: str, *, write_ack_ms: float, consolidation_bg_ms: float, read_ms: float) -> Dict[str, Any]:
    """Chelated EK5 v3 mutates on the write path before any read, so both horizons match."""
    return {
        "delta0": context,
        "settled": context,
        "task_done_at_delta0": True,
        "task_done_after_settle": True,
        "write_ack_ms": float(write_ack_ms),
        "consolidation_bg_ms": float(consolidation_bg_ms),
        "settled_read_ms": float(read_ms),
        "synchronous_ek5_fence": True,
    }


def format_mem0_results(found: Any) -> str:
    if isinstance(found, Mapping):
        results = list(found.get("results") or [])
    elif found is None:
        results = []
    else:
        results = list(found)
    if not results:
        return "(No memories found.)"
    lines: List[str] = []
    for row in results:
        if isinstance(row, Mapping):
            lines.append(f"- {row.get('memory', '')}")
        else:
            lines.append(f"- {row}")
    return "\n".join(lines)


def apply_horizon_grades(
    row: Dict[str, Any],
    graded_settled: Mapping[str, Mapping[str, Any]],
    graded_delta0: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    for arm, grade in graded_settled.items():
        slot = row["arms"][arm]
        slot["judge_settled"] = dict(grade)
        slot["correct_settled"] = bool(grade.get("correct"))
        audit = slot.get("graph_audit_settled") or {}
        if audit.get("status") == "ARM_FAILURE" or slot.get("arm_failure"):
            slot["arm_failure"] = True
            slot["correct_settled"] = False
    for arm, grade in graded_delta0.items():
        slot = row["arms"][arm]
        slot["judge_delta0s"] = dict(grade)
        slot["correct_delta0s"] = bool(grade.get("correct"))
    return row


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize_suite_b(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Descriptive temporal accuracy and token burn. Not pooled with Suite A and not a Holm family."""
    arms: Dict[str, Any] = {}
    for arm in ARENA_ARMS:
        arms[arm] = {
            "n": len(rows),
            "temporal_accuracy_settled": _mean([1.0 if row["arms"][arm]["correct_settled"] else 0.0 for row in rows]),
            "temporal_accuracy_delta0s": _mean([1.0 if row["arms"][arm]["correct_delta0s"] else 0.0 for row in rows]),
            "mean_write_prompt_tokens": _mean([float(row["arms"][arm]["write_prompt_tokens"]) for row in rows]),
            "mean_write_completion_tokens": _mean([float(row["arms"][arm]["write_completion_tokens"]) for row in rows]),
            "mean_read_prompt_tokens": _mean([float(row["arms"][arm]["read_prompt_tokens"]) for row in rows]),
            "mean_read_completion_tokens": _mean([float(row["arms"][arm]["read_completion_tokens"]) for row in rows]),
        }
    return {
        "suite": "Suite_B_LongMemEval_S_Knowledge_Update_N20",
        "label": "descriptive-only",
        "unpooled": True,
        "inferential_claim": False,
        "holm_bonferroni": None,
        "n_items": len(rows),
        "arms": arms,
    }


def suite_a_horizon_inference(
    rows: Sequence[Mapping[str, Any]],
    horizon: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Paired bootstrap, Tango 95%, Clopper-Pearson 95%, and Holm-Bonferroni for one horizon.

    The Holm family is exactly Chelated−Graphiti and Chelated−Mem0 on that horizon.
    Abstention rows (``U``) stay in the N=60 primary scores and are also reported
    in a separate block that is not added to the Holm family. ``n<40`` uses the
    weak monotonicity already enforced by ``paired_bootstrap_wave6``.
    """
    if horizon not in {"delta0", "settled"}:
        raise ValueError(f"unknown horizon {horizon!r}")
    from wave6_common import holm_bonferroni_stepdown, paired_bootstrap_wave6

    field = "correct_delta0s" if horizon == "delta0" else "correct_settled"
    item_ids = [str(row["item_id"]) for row in rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("duplicate Suite A item ids")
    score_maps: Dict[str, Dict[str, float]] = {}
    for arm in ARENA_ARMS:
        score_maps[arm] = {str(row["item_id"]): 1.0 if row["arms"][arm][field] else 0.0 for row in rows}
    comparisons: Dict[str, Any] = {}
    p_values: Dict[str, float] = {}
    for other in (ARM_GRAPHITI, ARM_MEM0):
        name = f"{ARM_CHELATED}_minus_{other}"
        boot = paired_bootstrap_wave6(
            score_maps[ARM_CHELATED],
            score_maps[other],
            n_bootstrap=n_bootstrap,
            seed=seed,
            expected_ids=item_ids,
        )
        comparisons[name] = boot
        p_values[name] = float(boot["p_value_two_sided"])
    holm = holm_bonferroni_stepdown(p_values)
    stale_rows = [row for row in rows if row.get("stale_values")]
    context_key = "stale_context_delta0s" if horizon == "delta0" else "stale_context_settled"
    answer_key = "stale_answer_delta0s" if horizon == "delta0" else "stale_answer_settled"
    stale: Dict[str, Any] = {}
    for arm in ARENA_ARMS:
        stale[arm] = {
            "context_substring_rate": _mean([1.0 if row["arms"][arm].get(context_key) else 0.0 for row in stale_rows]),
            "answer_stale_rate": _mean([1.0 if row["arms"][arm].get(answer_key) else 0.0 for row in stale_rows]),
            "n_stale_items": len(stale_rows),
        }
    u_rows = [row for row in rows if row.get("abstention_required")]
    u_block: Optional[Dict[str, Any]] = None
    if u_rows:
        u_ids = [str(row["item_id"]) for row in u_rows]
        u_maps = {
            arm: {str(row["item_id"]): 1.0 if row["arms"][arm][field] else 0.0 for row in u_rows}
            for arm in ARENA_ARMS
        }
        u_block = {
            "n": len(u_rows),
            "in_holm_family": False,
            "monotonicity": "weak" if len(u_rows) < 40 else "strict",
            "accuracies": {arm: _mean(list(u_maps[arm].values())) for arm in ARENA_ARMS},
        }
        if len(u_rows) >= 2:
            u_block["paired"] = {
                other: paired_bootstrap_wave6(
                    u_maps[ARM_CHELATED],
                    u_maps[other],
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    expected_ids=u_ids,
                )
                for other in (ARM_GRAPHITI, ARM_MEM0)
            }
    clocks: Dict[str, Any] = {}
    for arm in ARENA_ARMS:
        clocks[arm] = {
            "accuracy": _mean(list(score_maps[arm].values())),
            "write_ack_p50_ms": _quantile([float(row["arms"][arm]["write_ack_ms"]) for row in rows], 0.50),
            "consolidation_bg_mean_ms": _mean([float(row["arms"][arm]["consolidation_bg_ms"]) for row in rows]),
            "mean_write_prompt_tokens": _mean([float(row["arms"][arm]["write_prompt_tokens"]) for row in rows]),
            "mean_write_completion_tokens": _mean([float(row["arms"][arm]["write_completion_tokens"]) for row in rows]),
            "mean_read_prompt_tokens": _mean([float(row["arms"][arm]["read_prompt_tokens"]) for row in rows]),
            "mean_read_completion_tokens": _mean([float(row["arms"][arm]["read_completion_tokens"]) for row in rows]),
            "arm_failure_count": int(sum(1 for row in rows if row["arms"][arm].get("arm_failure"))),
        }
    return {
        "horizon": horizon,
        "n_items": len(rows),
        "primary_arm": ARM_CHELATED,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "comparisons": comparisons,
        "holm_bonferroni": holm,
        "U_separate": u_block,
        "stale_leakage": stale,
        "arms": clocks,
    }


def call_nemotron_reader(
    endpoint: str,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 128,
    post: Optional[Callable[..., Any]] = None,
) -> Tuple[str, int, int, float]:
    """Identical Nemotron reader for every arm. Missing usage fails closed."""
    import requests

    poster = post or requests.post
    url = f"{str(endpoint).rstrip('/')}/chat/completions"
    started = time.perf_counter()
    last_error: Optional[BaseException] = None
    for attempt in range(5):
        try:
            response = poster(
                url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": int(max_tokens),
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=180.0,
            )
            status = int(getattr(response, "status_code", 200))
            if status in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {status}: {getattr(response, 'text', '')[:160]}")
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {getattr(response, 'text', '')[:160]}")
            body = response.json()
            usage = body.get("usage")
            if not isinstance(usage, dict) or "prompt_tokens" not in usage or "completion_tokens" not in usage:
                raise RuntimeError("reader response.usage is missing prompt_tokens or completion_tokens")
            text = str((body.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
            wall_ms = (time.perf_counter() - started) * 1000.0
            return text, int(usage["prompt_tokens"]), int(usage["completion_tokens"]), wall_ms
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                raise
            time.sleep(0.01 if os.environ.get("WAVE11_FAST_RETRY") == "1" else 1.5 * (attempt + 1))
    raise RuntimeError(f"reader failed: {last_error}")
