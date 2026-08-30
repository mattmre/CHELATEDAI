"""Deterministic, learnable evaluation corpus with public functional oracles.

This module is deliberately incompatible with :mod:`egv.evaluation.dataset`'s
``egv-evaluation-data-v1`` corpus.  V1 generated answer material from an
evaluator secret.  In this V2 corpus the secret may select inputs, but the
public specification and runtime input are the complete, deterministic oracle.

The reference-source builder accepts a closed model-visible task record only.
It has no corpus, seed, hidden-case, or expected-output parameter.  The local
proof helper uses ``LocalTestSandbox`` as a test helper; it is evidence of
solvability, never production sandbox acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from ..canonical import canonical_json, canonical_value, digest_bytes, digest_for
from .errors import EvaluationError
from .sandbox import LocalTestSandbox, validate_candidate_source_contract


LEARNABLE_DATA_VERSION = "egv-learnable-data-v2"
LEARNABLE_TASK_VERSION = "egv-learnable-task-v2"
LEARNABLE_SPEC_VERSION = "egv-learnable-public-spec-v2"
LEARNABLE_DSL_VERSION = "egv-public-expression-dsl-v1"
LEARNABLE_PRIVATE_VERSION = "egv-learnable-private-v2"
LEARNABLE_MODEL_PACKAGE_VERSION = "egv-learnable-model-package-v2"
LEARNABLE_LOCAL_PROOF_VERSION = "egv-learnable-local-proof-v2"
INCOMPATIBLE_V1_VERSION = "egv-evaluation-data-v1"
LEARNABLE_SEED_ENV = "EGV_LEARNABLE_V2_SEED_FILE"
LEARNABLE_SEED_BYTES = 32
SPLITS_V2 = ("train", "dev", "heldout", "blind")
HIDDEN_CASES_PER_TASK = 2


@dataclass(frozen=True)
class LearnableFamilySpec:
    """Frozen task allocation for one public functional family."""

    family_id: str
    train: int
    dev: int
    heldout: int
    blind: int
    problem: str

    def count(self, split: str) -> int:
        if split == "train":
            return self.train
        if split == "dev":
            return self.dev
        if split == "heldout":
            return self.heldout
        if split == "blind":
            return self.blind
        raise ValueError("unknown learnable-v2 split")


FAMILY_SPECS_V2 = (
    LearnableFamilySpec("PURE_FUNCTION", 4, 2, 20, 3, "apply a public deterministic arithmetic function"),
    LearnableFamilySpec("PARSER_EDGE", 4, 2, 20, 3, "preserve empty tokens at a public delimiter edge"),
    LearnableFamilySpec("STATE_TRANSITION", 3, 2, 20, 3, "apply a public finite-state transition"),
    LearnableFamilySpec("DATA_TRANSFORM", 3, 2, 20, 3, "sort and transform rows under a public schema"),
    LearnableFamilySpec("RESOURCE_BOUND", 3, 2, 20, 3, "aggregate a bounded list in one public pass"),
    LearnableFamilySpec("DEPENDENCY_CONTRACT", 3, 2, 20, 3, "honor a public dependency lookup contract"),
)
FAMILY_BY_ID_V2 = {spec.family_id: spec for spec in FAMILY_SPECS_V2}


def _task_id(family_id: str, split: str, ordinal: int) -> str:
    family_slug = family_id.lower().replace("_", "-")
    return "egv-learnable-{}-{}-{:03d}-v2".format(family_slug, split, ordinal)


STAGE_A_TASK_IDS = tuple(_task_id(family.family_id, "train", 1) for family in FAMILY_SPECS_V2)


_PUBLIC_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "incompatible_with",
        "task_id",
        "family_id",
        "split",
        "ordinal",
        "public_spec",
        "public_spec_digest",
        "starter_source",
        "starter_source_digest",
        "record_digest",
    }
)
_PUBLIC_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "incompatible_with",
        "task_count",
        "split_counts",
        "family_counts",
        "hidden_cases_per_task",
        "stage_a_task_ids",
        "tasks",
        "manifest_digest",
    }
)
_MODEL_PACKAGE_FIELDS = frozenset({"schema_version", "task", "package_digest"})
_PRIVATE_MANIFEST_FIELDS = frozenset(
    {"schema_version", "public_manifest_digest", "seed_digest", "tasks", "manifest_digest"}
)
_PRIVATE_TASK_FIELDS = frozenset({"task_id", "cases"})
_PRIVATE_CASE_FIELDS = frozenset(
    {"case_id", "ordinal", "input_digest", "expected_output_digest"}
)


def _require_closed(record: Mapping[str, Any], fields: Set[str], label: str) -> None:
    if not isinstance(record, Mapping) or set(record) != fields:
        raise ValueError("{} is not a closed record".format(label))


def _digest_without(record: Mapping[str, Any], field: str) -> str:
    return digest_for({key: value for key, value in record.items() if key != field})


def _public_material(task_id: str) -> bytes:
    return hashlib.sha256(("egv-learnable-public-v2|" + task_id).encode("utf-8")).digest()


def _literal(value: Any) -> Dict[str, Any]:
    return {"op": "literal", "value": value}


def _input(*path: str) -> Dict[str, Any]:
    return {"op": "input", "path": list(path)}


def _variable(name: str, *path: str) -> Dict[str, Any]:
    return {"op": "variable", "name": name, "path": list(path)}


def _object(**fields: Any) -> Dict[str, Any]:
    return {
        "op": "object",
        "fields": [{"name": name, "value": value} for name, value in fields.items()],
    }


def _public_spec(task_id: str, family_id: str) -> Dict[str, Any]:
    material = _public_material(task_id)
    common = {
        "schema_version": LEARNABLE_SPEC_VERSION,
        "expression_language": LEARNABLE_DSL_VERSION,
        "family_id": family_id,
        "task_scope": task_id,
        "input_contract": {
            "closed_fields": ["case_nonce", "payload", "task_scope"],
            "case_nonce": {"type": "text", "length": 20},
            "task_scope_must_match": True,
        },
    }
    if family_id == "PURE_FUNCTION":
        bias = 5 + material[0] % 19
        common.update(
            {
                "parameters": {"bias": bias},
                "payload_contract": {"type": "integer", "exclude_boolean": True},
                "output_contract": ["case_nonce", "task_scope", "value"],
                "program": _object(
                    case_nonce=_input("case_nonce"),
                    task_scope=_input("task_scope"),
                    value={
                        "op": "add_integers",
                        "items": [_input("payload"), _input("payload"), _literal(bias)],
                    },
                ),
            }
        )
    elif family_id == "PARSER_EDGE":
        delimiters = ("::", "||", "~|~")
        delimiter = delimiters[material[0] % len(delimiters)]
        common.update(
            {
                "parameters": {"delimiter": delimiter},
                "payload_contract": {"type": "text"},
                "output_contract": ["case_nonce", "task_scope", "tokens"],
                "program": _object(
                    case_nonce=_input("case_nonce"),
                    task_scope=_input("task_scope"),
                    tokens={
                        "op": "split_text",
                        "text": _input("payload"),
                        "delimiter": _literal(delimiter),
                    },
                ),
            }
        )
    elif family_id == "STATE_TRANSITION":
        advance_by = 1 + material[0] % 7
        common.update(
            {
                "parameters": {"advance_by": advance_by},
                "payload_contract": {
                    "type": "closed_object",
                    "fields": {
                        "event": {"type": "enum", "values": ["reset", "advance"]},
                        "state": {"type": "integer", "exclude_boolean": True},
                    },
                },
                "output_contract": ["case_nonce", "state", "task_scope"],
                "program": _object(
                    case_nonce=_input("case_nonce"),
                    state={
                        "op": "if_equal",
                        "left": _input("payload", "event"),
                        "right": _literal("reset"),
                        "then": _literal(0),
                        "else": {
                            "op": "add_integers",
                            "items": [_input("payload", "state"), _literal(advance_by)],
                        },
                    },
                    task_scope=_input("task_scope"),
                ),
            }
        )
    elif family_id == "DATA_TRANSFORM":
        common.update(
            {
                "parameters": {"sort_key": "key"},
                "payload_contract": {
                    "type": "non_empty_list",
                    "items": {
                        "type": "closed_object",
                        "fields": {
                            "key": {"type": "text"},
                            "value": {"type": "integer", "exclude_boolean": True},
                        },
                    },
                },
                "output_contract": ["case_nonce", "rows", "task_scope"],
                "program": _object(
                    case_nonce=_input("case_nonce"),
                    rows={
                        "op": "map_list",
                        "items": {
                            "op": "sort_objects",
                            "items": _input("payload"),
                            "key": "key",
                        },
                        "variable": "row",
                        "body": _object(
                            key=_variable("row", "key"),
                            value={
                                "op": "add_integers",
                                "items": [
                                    _variable("row", "value"),
                                    _variable("row", "value"),
                                ],
                            },
                        ),
                    },
                    task_scope=_input("task_scope"),
                ),
            }
        )
    elif family_id == "RESOURCE_BOUND":
        max_items = 16 + material[0] % 17
        common.update(
            {
                "parameters": {"max_items": max_items},
                "payload_contract": {
                    "type": "list",
                    "max_items": max_items,
                    "items": {"type": "integer", "exclude_boolean": True},
                },
                "output_contract": ["case_nonce", "count", "sum", "task_scope"],
                "program": _object(
                    case_nonce=_input("case_nonce"),
                    count={"op": "list_length", "items": _input("payload")},
                    sum={"op": "sum_integers", "items": _input("payload")},
                    task_scope=_input("task_scope"),
                ),
            }
        )
    elif family_id == "DEPENDENCY_CONTRACT":
        default = "missing:" + task_id
        common.update(
            {
                "parameters": {"default": default},
                "payload_contract": {
                    "type": "closed_object",
                    "fields": {
                        "dependencies": {
                            "type": "mapping",
                            "key_type": "text",
                            "value_type": "json",
                        },
                        "key": {"type": "text"},
                    },
                },
                "output_contract": ["case_nonce", "task_scope", "value"],
                "program": _object(
                    case_nonce=_input("case_nonce"),
                    task_scope=_input("task_scope"),
                    value={
                        "op": "mapping_get",
                        "mapping": _input("payload", "dependencies"),
                        "key": _input("payload", "key"),
                        "default": _literal(default),
                    },
                ),
            }
        )
    else:
        raise ValueError("unknown learnable-v2 family")
    return common


def _validate_runtime_input(spec: Mapping[str, Any], runtime_input: Any) -> Mapping[str, Any]:
    if not isinstance(runtime_input, Mapping):
        raise ValueError("runtime input must be an object")
    if set(runtime_input) != {"case_nonce", "payload", "task_scope"}:
        raise ValueError("runtime input is not closed")
    if runtime_input.get("task_scope") != spec.get("task_scope"):
        raise ValueError("runtime input task scope does not match public spec")
    nonce = runtime_input.get("case_nonce")
    if not isinstance(nonce, str) or len(nonce) != 20:
        raise ValueError("runtime input case nonce is invalid")
    payload = runtime_input["payload"]
    family_id = spec.get("family_id")
    if family_id == "PURE_FUNCTION":
        if not isinstance(payload, int) or isinstance(payload, bool):
            raise ValueError("PURE_FUNCTION payload must be an integer")
    elif family_id == "PARSER_EDGE":
        if not isinstance(payload, str):
            raise ValueError("PARSER_EDGE payload must be text")
    elif family_id == "STATE_TRANSITION":
        if not isinstance(payload, Mapping) or set(payload) != {"event", "state"}:
            raise ValueError("STATE_TRANSITION payload is invalid")
        if payload["event"] not in {"reset", "advance"}:
            raise ValueError("STATE_TRANSITION event is invalid")
        if not isinstance(payload["state"], int) or isinstance(payload["state"], bool):
            raise ValueError("STATE_TRANSITION state must be an integer")
    elif family_id == "DATA_TRANSFORM":
        if not isinstance(payload, list) or not payload:
            raise ValueError("DATA_TRANSFORM payload must be a non-empty list")
        for row in payload:
            if not isinstance(row, Mapping) or set(row) != {"key", "value"}:
                raise ValueError("DATA_TRANSFORM row is invalid")
            if not isinstance(row["key"], str):
                raise ValueError("DATA_TRANSFORM row key must be text")
            if not isinstance(row["value"], int) or isinstance(row["value"], bool):
                raise ValueError("DATA_TRANSFORM row value must be an integer")
    elif family_id == "RESOURCE_BOUND":
        max_items = spec["payload_contract"]["max_items"]
        if not isinstance(payload, list) or len(payload) > max_items:
            raise ValueError("RESOURCE_BOUND payload exceeds the public bound")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in payload):
            raise ValueError("RESOURCE_BOUND items must be integers")
    elif family_id == "DEPENDENCY_CONTRACT":
        if not isinstance(payload, Mapping) or set(payload) != {"dependencies", "key"}:
            raise ValueError("DEPENDENCY_CONTRACT payload is invalid")
        dependencies = payload["dependencies"]
        if not isinstance(dependencies, Mapping) or not all(
            isinstance(key, str) for key in dependencies
        ):
            raise ValueError("DEPENDENCY_CONTRACT dependencies must be a text-keyed mapping")
        if not isinstance(payload["key"], str):
            raise ValueError("DEPENDENCY_CONTRACT key must be text")
    else:
        raise ValueError("unknown learnable-v2 family")
    return runtime_input


_EXPRESSION_FIELDS = {
    "literal": {"op", "value"},
    "input": {"op", "path"},
    "variable": {"op", "name", "path"},
    "object": {"op", "fields"},
    "add_integers": {"op", "items"},
    "split_text": {"op", "text", "delimiter"},
    "if_equal": {"op", "left", "right", "then", "else"},
    "sort_objects": {"op", "items", "key"},
    "map_list": {"op", "items", "variable", "body"},
    "list_length": {"op", "items"},
    "sum_integers": {"op", "items"},
    "mapping_get": {"op", "mapping", "key", "default"},
}


def _expression_path(root: Any, path: Any) -> Any:
    if not isinstance(path, list) or any(not isinstance(part, str) or not part for part in path):
        raise ValueError("public expression path is invalid")
    current = root
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError("public expression path is unavailable")
        current = current[part]
    return current


def _evaluate_expression(node: Any, runtime_input: Mapping[str, Any], variables: Mapping[str, Any]) -> Any:
    if not isinstance(node, Mapping):
        raise ValueError("public expression node must be an object")
    op = node.get("op")
    if op not in _EXPRESSION_FIELDS or set(node) != _EXPRESSION_FIELDS[op]:
        raise ValueError("public expression node is not closed or uses an unknown operator")
    if op == "literal":
        return canonical_value(node["value"])
    if op == "input":
        return _expression_path(runtime_input, node["path"])
    if op == "variable":
        name = node["name"]
        if not isinstance(name, str) or name not in variables:
            raise ValueError("public expression variable is unavailable")
        return _expression_path(variables[name], node["path"])
    if op == "object":
        fields = node["fields"]
        if not isinstance(fields, list):
            raise ValueError("public object fields must be a list")
        result: Dict[str, Any] = {}
        for field in fields:
            if not isinstance(field, Mapping) or set(field) != {"name", "value"}:
                raise ValueError("public object field is not closed")
            name = field["name"]
            if not isinstance(name, str) or not name or name in result:
                raise ValueError("public object field name is invalid or duplicated")
            result[name] = _evaluate_expression(field["value"], runtime_input, variables)
        return result
    if op == "add_integers":
        items = node["items"]
        if not isinstance(items, list) or not items:
            raise ValueError("add_integers requires a non-empty item list")
        values = [_evaluate_expression(item, runtime_input, variables) for item in items]
        if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
            raise ValueError("add_integers operands must be integers")
        return sum(values)
    if op == "split_text":
        text = _evaluate_expression(node["text"], runtime_input, variables)
        delimiter = _evaluate_expression(node["delimiter"], runtime_input, variables)
        if not isinstance(text, str) or not isinstance(delimiter, str) or not delimiter:
            raise ValueError("split_text operands must be text with a non-empty delimiter")
        return text.split(delimiter)
    if op == "if_equal":
        left = _evaluate_expression(node["left"], runtime_input, variables)
        right = _evaluate_expression(node["right"], runtime_input, variables)
        branch = node["then"] if left == right else node["else"]
        return _evaluate_expression(branch, runtime_input, variables)
    if op == "sort_objects":
        items = _evaluate_expression(node["items"], runtime_input, variables)
        key = node["key"]
        if not isinstance(items, list) or not isinstance(key, str) or not key:
            raise ValueError("sort_objects operands are invalid")
        if any(not isinstance(item, Mapping) or key not in item for item in items):
            raise ValueError("sort_objects item lacks the public key")
        return sorted(items, key=lambda item: item[key])
    if op == "map_list":
        items = _evaluate_expression(node["items"], runtime_input, variables)
        variable = node["variable"]
        if not isinstance(items, list) or not isinstance(variable, str) or not variable:
            raise ValueError("map_list operands are invalid")
        return [
            _evaluate_expression(node["body"], runtime_input, {**variables, variable: item})
            for item in items
        ]
    if op == "list_length":
        items = _evaluate_expression(node["items"], runtime_input, variables)
        if not isinstance(items, list):
            raise ValueError("list_length operand must be a list")
        return len(items)
    if op == "sum_integers":
        items = _evaluate_expression(node["items"], runtime_input, variables)
        if not isinstance(items, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in items
        ):
            raise ValueError("sum_integers operand must be an integer list")
        return sum(items)
    if op == "mapping_get":
        mapping = _evaluate_expression(node["mapping"], runtime_input, variables)
        key = _evaluate_expression(node["key"], runtime_input, variables)
        default = _evaluate_expression(node["default"], runtime_input, variables)
        if not isinstance(mapping, Mapping):
            raise ValueError("mapping_get mapping operand must be an object")
        return mapping.get(key, default)
    raise AssertionError("validated public expression operator was not handled")


def solve_public_spec(public_spec: Mapping[str, Any], runtime_input: Any) -> Any:
    """Return the unique answer using public spec and runtime input only."""

    if not isinstance(public_spec, Mapping):
        raise ValueError("public spec must be an object")
    family_id = public_spec.get("family_id")
    task_scope = public_spec.get("task_scope")
    if not isinstance(task_scope, str) or public_spec != _public_spec(task_scope, str(family_id)):
        raise ValueError("public spec differs from the frozen functional contract")
    value = _validate_runtime_input(public_spec, runtime_input)
    return canonical_value(_evaluate_expression(public_spec["program"], value, {}))


def _compile_expression(node: Mapping[str, Any], variables: Mapping[str, str]) -> str:
    op = node["op"]
    if op == "literal":
        return repr(node["value"])
    if op == "input":
        return "value" + "".join("[{!r}]".format(part) for part in node["path"])
    if op == "variable":
        name = variables[node["name"]]
        return name + "".join("[{!r}]".format(part) for part in node["path"])
    if op == "object":
        return "{" + ", ".join(
            "{!r}: {}".format(field["name"], _compile_expression(field["value"], variables))
            for field in node["fields"]
        ) + "}"
    if op == "add_integers":
        return "(" + " + ".join(_compile_expression(item, variables) for item in node["items"]) + ")"
    if op == "split_text":
        return "{}.split({})".format(
            _compile_expression(node["text"], variables),
            _compile_expression(node["delimiter"], variables),
        )
    if op == "if_equal":
        return "({} if {} == {} else {})".format(
            _compile_expression(node["then"], variables),
            _compile_expression(node["left"], variables),
            _compile_expression(node["right"], variables),
            _compile_expression(node["else"], variables),
        )
    if op == "sort_objects":
        return "sorted({}, key=lambda _sort_item: _sort_item[{!r}])".format(
            _compile_expression(node["items"], variables), node["key"]
        )
    if op == "map_list":
        variable = node["variable"]
        python_name = "_map_{}".format(variable)
        return "[{} for {} in {}]".format(
            _compile_expression(node["body"], {**variables, variable: python_name}),
            python_name,
            _compile_expression(node["items"], variables),
        )
    if op == "list_length":
        return "len({})".format(_compile_expression(node["items"], variables))
    if op == "sum_integers":
        return "sum({})".format(_compile_expression(node["items"], variables))
    if op == "mapping_get":
        return "{}.get({}, {})".format(
            _compile_expression(node["mapping"], variables),
            _compile_expression(node["key"], variables),
            _compile_expression(node["default"], variables),
        )
    raise ValueError("unknown public expression operator")


def _reference_source_for_spec(public_spec: Mapping[str, Any]) -> bytes:
    scope = public_spec["task_scope"]
    source_text = (
        "def main(value):\n"
        "    if value.get('task_scope') != {!r}:\n"
        "        return {{'error': 'wrong-task'}}\n"
        "    return {}\n"
    ).format(scope, _compile_expression(public_spec["program"], {}))
    source = source_text.encode("utf-8")
    violation = validate_candidate_source_contract(source)
    if violation is not None:
        raise ValueError("reference source violates pure-return-v1: {}".format(violation))
    return source


_STARTER_SOURCE = (
    "def main(value):\n"
    "    return {'case_nonce': value.get('case_nonce'), "
    "'task_scope': value.get('task_scope'), 'status': 'unimplemented'}\n"
)


def _record_core(task_id: str, family_id: str, split: str, ordinal: int) -> Dict[str, Any]:
    spec = _public_spec(task_id, family_id)
    starter_source = _STARTER_SOURCE
    record: Dict[str, Any] = {
        "schema_version": LEARNABLE_TASK_VERSION,
        "incompatible_with": [INCOMPATIBLE_V1_VERSION],
        "task_id": task_id,
        "family_id": family_id,
        "split": split,
        "ordinal": ordinal,
        "public_spec": spec,
        "public_spec_digest": digest_for(spec),
        "starter_source": starter_source,
        "starter_source_digest": digest_bytes(starter_source.encode("utf-8")),
    }
    record["record_digest"] = digest_for(record)
    return record


def validate_public_task_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a closed, model-visible task record and all source bindings."""

    _require_closed(record, set(_PUBLIC_TASK_FIELDS), "learnable-v2 public task")
    if record.get("schema_version") == INCOMPATIBLE_V1_VERSION:
        raise ValueError("evaluation data v1 is explicitly incompatible with learnable v2")
    if record.get("schema_version") != LEARNABLE_TASK_VERSION:
        raise ValueError("unsupported learnable-v2 task schema")
    if record.get("incompatible_with") != [INCOMPATIBLE_V1_VERSION]:
        raise ValueError("learnable-v2 task must declare v1 incompatibility")
    family_id = record.get("family_id")
    split = record.get("split")
    ordinal = record.get("ordinal")
    if family_id not in FAMILY_BY_ID_V2 or split not in SPLITS_V2:
        raise ValueError("learnable-v2 task has unknown family or split")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 1 <= ordinal <= FAMILY_BY_ID_V2[family_id].count(split):
        raise ValueError("learnable-v2 task ordinal is outside the frozen allocation")
    expected_id = _task_id(family_id, split, ordinal)
    if record.get("task_id") != expected_id:
        raise ValueError("learnable-v2 task ID differs from frozen identity")
    expected_spec = _public_spec(expected_id, family_id)
    if record.get("public_spec") != expected_spec:
        raise ValueError("public spec differs from frozen task specification")
    if record.get("public_spec_digest") != digest_for(expected_spec):
        raise ValueError("public spec digest mismatch")
    starter = record.get("starter_source")
    if starter != _STARTER_SOURCE:
        raise ValueError("starter source differs from the frozen task specification")
    if record.get("starter_source_digest") != digest_bytes(_STARTER_SOURCE.encode("utf-8")):
        raise ValueError("starter source digest mismatch")
    starter_violation = validate_candidate_source_contract(starter.encode("utf-8"))
    if starter_violation is not None:
        raise ValueError("starter source violates pure-return-v1")
    if record.get("record_digest") != _digest_without(record, "record_digest"):
        raise ValueError("public task record digest mismatch")
    return canonical_value(record)


def build_reference_source(public_task_record: Mapping[str, Any]) -> bytes:
    """Build a solution from the closed public task record and nothing else."""

    record = validate_public_task_record(public_task_record)
    return _reference_source_for_spec(record["public_spec"])


def _read_secret_seed(secret_seed_file: Optional[Union[Path, str]]) -> bytes:
    configured = secret_seed_file or os.environ.get(LEARNABLE_SEED_ENV)
    if not configured:
        raise EvaluationError(
            "learnable-v2 generation requires an evaluator-only seed file; "
            "set {} or pass secret_seed_file".format(LEARNABLE_SEED_ENV)
        )
    try:
        secret = Path(configured).read_bytes()
    except OSError as exc:
        raise EvaluationError("learnable-v2 evaluator seed file is unavailable") from exc
    if len(secret) != LEARNABLE_SEED_BYTES:
        raise EvaluationError("learnable-v2 evaluator seed must contain exactly 32 bytes")
    return secret


def _derive(secret: bytes, label: str) -> str:
    return hmac.new(secret, label.encode("utf-8"), hashlib.sha256).hexdigest()


def _hidden_input(secret: bytes, record: Mapping[str, Any], case_ordinal: int) -> Dict[str, Any]:
    task_id = record["task_id"]
    family_id = record["family_id"]
    spec = record["public_spec"]
    prefix = "egv-learnable-v2|{}|case-{}".format(task_id, case_ordinal)
    nonce = _derive(secret, prefix + "|nonce")[:20]
    material = _derive(secret, prefix + "|payload")
    if family_id == "PURE_FUNCTION":
        payload: Any = 100 + int(material[:8], 16) % 10000
    elif family_id == "PARSER_EDGE":
        parts = ["p" + material[0:8], "", "p" + material[8:16], "p" + material[16:24]]
        payload = spec["parameters"]["delimiter"].join(parts)
    elif family_id == "STATE_TRANSITION":
        payload = {"event": "reset" if int(material[0], 16) % 2 == 0 else "advance", "state": int(material[1:7], 16) % 100}
    elif family_id == "DATA_TRANSFORM":
        payload = [
            {"key": "k{}{}".format(index, material[index * 6 : (index + 1) * 6]), "value": int(material[24 + index * 4 : 28 + index * 4], 16) % 1000}
            for index in range(3)
        ]
        payload = [payload[2], payload[0], payload[1]]
    elif family_id == "RESOURCE_BOUND":
        length = 5 + int(material[0], 16) % 5
        payload = [int(material[index * 4 : (index + 1) * 4], 16) % 1000 for index in range(length)]
    elif family_id == "DEPENDENCY_CONTRACT":
        first_key = "dep-" + material[0:8]
        second_key = "dep-" + material[8:16]
        dependencies = {first_key: "value-" + material[16:24], second_key: "value-" + material[24:32]}
        key = first_key if case_ordinal % 2 else "missing-" + material[32:40]
        payload = {"dependencies": dependencies, "key": key}
    else:  # pragma: no cover - frozen family validation reaches this first
        raise ValueError("unknown learnable-v2 family")
    return {"case_nonce": nonce, "payload": payload, "task_scope": task_id}


@dataclass(frozen=True)
class LearnableCase:
    case_id: str
    ordinal: int
    input_value: Any
    expected_output: Any
    input_digest: str
    expected_output_digest: str

    def private_record(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "input_digest": self.input_digest,
            "expected_output_digest": self.expected_output_digest,
        }


@dataclass(frozen=True)
class LearnableTask:
    public_record: Mapping[str, Any]
    hidden_cases: Tuple[LearnableCase, ...]

    @property
    def task_id(self) -> str:
        return str(self.public_record["task_id"])

    @property
    def family_id(self) -> str:
        return str(self.public_record["family_id"])

    @property
    def split(self) -> str:
        return str(self.public_record["split"])

    @property
    def ordinal(self) -> int:
        return int(self.public_record["ordinal"])


def _build_task(secret: bytes, family: LearnableFamilySpec, split: str, ordinal: int) -> LearnableTask:
    task_id = _task_id(family.family_id, split, ordinal)
    public_record = _record_core(task_id, family.family_id, split, ordinal)
    hidden_cases: List[LearnableCase] = []
    for case_ordinal in range(1, HIDDEN_CASES_PER_TASK + 1):
        input_value = _hidden_input(secret, public_record, case_ordinal)
        expected = solve_public_spec(public_record["public_spec"], input_value)
        input_digest = digest_for(input_value)
        expected_digest = digest_for(expected)
        case_id = "case_{}".format(
            digest_for({"task_id": task_id, "ordinal": case_ordinal, "input_digest": input_digest})
        )
        hidden_cases.append(
            LearnableCase(
                case_id=case_id,
                ordinal=case_ordinal,
                input_value=input_value,
                expected_output=expected,
                input_digest=input_digest,
                expected_output_digest=expected_digest,
            )
        )
    return LearnableTask(public_record=public_record, hidden_cases=tuple(hidden_cases))


def _strings(value: Any) -> Set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        found: Set[str] = set()
        for item in value.values():
            found.update(_strings(item))
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for item in value:
            found.update(_strings(item))
        return found
    return set()


class LearnableCorpusV2:
    """Frozen V2 corpus whose hidden targets are public functions of inputs."""

    def __init__(self, tasks: Sequence[LearnableTask], secret_seed: bytes) -> None:
        self.tasks = tuple(tasks)
        self._by_id = {task.task_id: task for task in self.tasks}
        self._secret_seed = bytes(secret_seed)
        self.validate()

    @classmethod
    def generate(cls, *, secret_seed_file: Optional[Union[Path, str]] = None) -> "LearnableCorpusV2":
        secret = _read_secret_seed(secret_seed_file)
        tasks: List[LearnableTask] = []
        for family in FAMILY_SPECS_V2:
            for split in SPLITS_V2:
                for ordinal in range(1, family.count(split) + 1):
                    tasks.append(_build_task(secret, family, split, ordinal))
        return cls(tasks, secret)

    @property
    def secret_seed_digest(self) -> str:
        return digest_bytes(self._secret_seed)

    def split(self, split: str) -> Tuple[LearnableTask, ...]:
        if split not in SPLITS_V2:
            raise ValueError("unknown learnable-v2 split")
        return tuple(task for task in self.tasks if task.split == split)

    def get(self, task_id: str) -> LearnableTask:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise KeyError("unknown learnable-v2 task ID") from exc

    def stage_a_tasks(self) -> Tuple[LearnableTask, ...]:
        """Return exactly one train task per family in the frozen order."""

        return tuple(self.get(task_id) for task_id in STAGE_A_TASK_IDS)

    def model_visible_package(self, task_id: str) -> Dict[str, Any]:
        task = self.get(task_id)
        package: Dict[str, Any] = {
            "schema_version": LEARNABLE_MODEL_PACKAGE_VERSION,
            "task": canonical_value(task.public_record),
        }
        package["package_digest"] = digest_for(package)
        return package

    def validate(self) -> None:
        if len(self._secret_seed) != LEARNABLE_SEED_BYTES:
            raise ValueError("learnable-v2 corpus has an invalid evaluator seed")
        expected_layout = [
            _task_id(family.family_id, split, ordinal)
            for family in FAMILY_SPECS_V2
            for split in SPLITS_V2
            for ordinal in range(1, family.count(split) + 1)
        ]
        actual_layout = [task.task_id for task in self.tasks]
        if actual_layout != expected_layout:
            raise ValueError("learnable-v2 IDs, ordering, or counts differ from the frozen contract")
        if len(set(actual_layout)) != len(actual_layout):
            raise ValueError("learnable-v2 corpus contains duplicate task IDs")
        for task in self.tasks:
            validate_public_task_record(task.public_record)
            if len(task.hidden_cases) != HIDDEN_CASES_PER_TASK:
                raise ValueError("learnable-v2 task has the wrong hidden-case count")
            expected_ordinals = list(range(1, HIDDEN_CASES_PER_TASK + 1))
            if [case.ordinal for case in task.hidden_cases] != expected_ordinals:
                raise ValueError("learnable-v2 hidden-case ordering differs from the frozen contract")
            source = build_reference_source(task.public_record)
            public_strings = _strings(task.public_record["public_spec"])
            for case in task.hidden_cases:
                regenerated_input = _hidden_input(
                    self._secret_seed, task.public_record, case.ordinal
                )
                if case.input_value != regenerated_input:
                    raise ValueError(
                        "learnable-v2 hidden input differs from evaluator-seed regeneration"
                    )
                regenerated_input_digest = digest_for(regenerated_input)
                if case.input_digest != regenerated_input_digest:
                    raise ValueError("learnable-v2 hidden input digest mismatch")
                expected_case_id = "case_{}".format(
                    digest_for(
                        {
                            "task_id": task.task_id,
                            "ordinal": case.ordinal,
                            "input_digest": regenerated_input_digest,
                        }
                    )
                )
                if case.case_id != expected_case_id:
                    raise ValueError("learnable-v2 hidden case ID mismatch")
                public_expected = solve_public_spec(
                    task.public_record["public_spec"], regenerated_input
                )
                if case.expected_output != public_expected:
                    raise ValueError("secret-dependent target rejected: expected output is not public-input derivable")
                if case.expected_output_digest != digest_for(public_expected):
                    raise ValueError("learnable-v2 expected-output digest mismatch")
                hidden_strings = (_strings(case.input_value) | _strings(case.expected_output)) - public_strings
                leaked = [token for token in hidden_strings if len(token) >= 8 and token.encode("utf-8") in source]
                if leaked:
                    raise ValueError("reference source contains a hidden-case or answer literal")
        self.validate_split_disjointness()

    def validate_split_disjointness(self) -> Dict[str, Any]:
        records: Dict[str, Dict[str, Set[str]]] = {}
        for split in SPLITS_V2:
            records[split] = {
                "case_ids": {case.case_id for task in self.split(split) for case in task.hidden_cases},
                "input_digests": {case.input_digest for task in self.split(split) for case in task.hidden_cases},
                "case_nonces": {
                    str(case.input_value["case_nonce"]) for task in self.split(split) for case in task.hidden_cases
                },
            }
        split_pairs: List[str] = []
        for left_index, left in enumerate(SPLITS_V2):
            for right in SPLITS_V2[left_index + 1 :]:
                split_pairs.append("{}:{}".format(left, right))
                for field in ("case_ids", "input_digests", "case_nonces"):
                    if records[left][field] & records[right][field]:
                        raise ValueError("learnable-v2 split overlap in {} for {} and {}".format(field, left, right))
        return {
            "pairwise_disjoint": True,
            "checked_fields": ["case_ids", "input_digests", "case_nonces"],
            "split_pairs": split_pairs,
        }

    def public_manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {
            "schema_version": LEARNABLE_DATA_VERSION,
            "incompatible_with": [INCOMPATIBLE_V1_VERSION],
            "task_count": len(self.tasks),
            "split_counts": {split: len(self.split(split)) for split in SPLITS_V2},
            "family_counts": {
                family.family_id: {split: family.count(split) for split in SPLITS_V2}
                for family in FAMILY_SPECS_V2
            },
            "hidden_cases_per_task": HIDDEN_CASES_PER_TASK,
            "stage_a_task_ids": list(STAGE_A_TASK_IDS),
            "tasks": [canonical_value(task.public_record) for task in self.tasks],
        }
        manifest["manifest_digest"] = digest_for(manifest)
        return manifest

    def private_manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {
            "schema_version": LEARNABLE_PRIVATE_VERSION,
            "public_manifest_digest": self.public_manifest()["manifest_digest"],
            "seed_digest": self.secret_seed_digest,
            "tasks": [
                {"task_id": task.task_id, "cases": [case.private_record() for case in task.hidden_cases]}
                for task in self.tasks
            ],
        }
        manifest["manifest_digest"] = digest_for(manifest)
        return manifest


def validate_model_visible_package(package: Mapping[str, Any]) -> Dict[str, Any]:
    _require_closed(package, set(_MODEL_PACKAGE_FIELDS), "learnable-v2 model package")
    if package.get("schema_version") != LEARNABLE_MODEL_PACKAGE_VERSION:
        raise ValueError("unsupported learnable-v2 model package")
    validate_public_task_record(package.get("task"))
    if package.get("package_digest") != _digest_without(package, "package_digest"):
        raise ValueError("learnable-v2 model package digest mismatch")
    forbidden = {"hidden_cases", "expected_output", "seed_digest", "seed_file_digest"}
    if forbidden & set(package):
        raise ValueError("learnable-v2 model package leaks evaluator-private fields")
    return canonical_value(package)


def validate_learnable_manifest(
    manifest: Mapping[str, Any], corpus: Optional[LearnableCorpusV2] = None
) -> Dict[str, Any]:
    """Validate the closed public manifest, rejecting V1 without conversion."""

    if isinstance(manifest, Mapping) and manifest.get("schema_version") == INCOMPATIBLE_V1_VERSION:
        raise ValueError("evaluation data v1 is explicitly incompatible with learnable v2")
    _require_closed(manifest, set(_PUBLIC_MANIFEST_FIELDS), "learnable-v2 public manifest")
    if manifest.get("schema_version") != LEARNABLE_DATA_VERSION:
        raise ValueError("unsupported learnable-v2 public manifest")
    if manifest.get("incompatible_with") != [INCOMPATIBLE_V1_VERSION]:
        raise ValueError("learnable-v2 manifest must declare v1 incompatibility")
    expected_split_counts = {"train": 20, "dev": 12, "heldout": 120, "blind": 18}
    expected_family_counts = {
        family.family_id: {split: family.count(split) for split in SPLITS_V2}
        for family in FAMILY_SPECS_V2
    }
    if manifest.get("task_count") != 170 or manifest.get("split_counts") != expected_split_counts:
        raise ValueError("learnable-v2 manifest task allocation mismatch")
    if manifest.get("family_counts") != expected_family_counts:
        raise ValueError("learnable-v2 manifest family allocation mismatch")
    if manifest.get("hidden_cases_per_task") != HIDDEN_CASES_PER_TASK:
        raise ValueError("learnable-v2 hidden-case count mismatch")
    if manifest.get("stage_a_task_ids") != list(STAGE_A_TASK_IDS):
        raise ValueError("learnable-v2 Stage-A selection differs from frozen order")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 170:
        raise ValueError("learnable-v2 manifest must enumerate exactly 170 tasks")
    for record in tasks:
        validate_public_task_record(record)
    expected_layout = [
        _task_id(family.family_id, split, ordinal)
        for family in FAMILY_SPECS_V2
        for split in SPLITS_V2
        for ordinal in range(1, family.count(split) + 1)
    ]
    if [record["task_id"] for record in tasks] != expected_layout:
        raise ValueError("learnable-v2 manifest task order or identities differ from the frozen layout")
    if manifest.get("manifest_digest") != _digest_without(manifest, "manifest_digest"):
        raise ValueError("learnable-v2 public manifest digest mismatch")
    if corpus is not None and manifest != corpus.public_manifest():
        raise ValueError("learnable-v2 manifest differs from the supplied corpus")
    return canonical_value(manifest)


def validate_learnable_private_manifest(
    manifest: Mapping[str, Any], corpus: Optional[LearnableCorpusV2] = None
) -> Dict[str, Any]:
    """Validate the closed digest-only evaluator manifest.

    The private manifest binds hidden cases but deliberately contains neither
    runtime inputs nor expected-output bytes.
    """

    _require_closed(manifest, set(_PRIVATE_MANIFEST_FIELDS), "learnable-v2 private manifest")
    if manifest.get("schema_version") != LEARNABLE_PRIVATE_VERSION:
        raise ValueError("unsupported learnable-v2 private manifest")
    for field in ("public_manifest_digest", "seed_digest", "manifest_digest"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("learnable-v2 private manifest has an invalid digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("learnable-v2 private manifest has a non-hex digest") from exc
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 170:
        raise ValueError("learnable-v2 private manifest must enumerate exactly 170 tasks")
    seen_task_ids: List[str] = []
    seen_case_ids: Set[str] = set()
    for task_record in tasks:
        _require_closed(task_record, set(_PRIVATE_TASK_FIELDS), "learnable-v2 private task")
        task_id = task_record.get("task_id")
        if not isinstance(task_id, str):
            raise ValueError("learnable-v2 private task ID is invalid")
        seen_task_ids.append(task_id)
        cases = task_record.get("cases")
        if not isinstance(cases, list) or len(cases) != HIDDEN_CASES_PER_TASK:
            raise ValueError("learnable-v2 private task has the wrong case count")
        for expected_ordinal, case in enumerate(cases, start=1):
            _require_closed(case, set(_PRIVATE_CASE_FIELDS), "learnable-v2 private case")
            if case.get("ordinal") != expected_ordinal:
                raise ValueError("learnable-v2 private case order mismatch")
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id.startswith("case_") or case_id in seen_case_ids:
                raise ValueError("learnable-v2 private case ID is invalid or duplicated")
            seen_case_ids.add(case_id)
            for field in ("input_digest", "expected_output_digest"):
                value = case.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError("learnable-v2 private case digest is invalid")
                try:
                    int(value, 16)
                except ValueError as exc:
                    raise ValueError("learnable-v2 private case digest is non-hex") from exc
    expected_layout = [
        _task_id(family.family_id, split, ordinal)
        for family in FAMILY_SPECS_V2
        for split in SPLITS_V2
        for ordinal in range(1, family.count(split) + 1)
    ]
    if seen_task_ids != expected_layout:
        raise ValueError("learnable-v2 private manifest task layout mismatch")
    if manifest.get("manifest_digest") != _digest_without(manifest, "manifest_digest"):
        raise ValueError("learnable-v2 private manifest digest mismatch")
    if corpus is not None and manifest != corpus.private_manifest():
        raise ValueError("learnable-v2 private manifest differs from the supplied corpus")
    return canonical_value(manifest)


def _semantic_local_result(sandbox: LocalTestSandbox, task: LearnableTask, case: LearnableCase, source: bytes, suffix: str) -> bool:
    result = sandbox.execute(
        source,
        case.input_value,
        artifact_digest=digest_bytes(source),
        candidate_id="{}-{}-{}".format(task.task_id, case.ordinal, suffix),
    )
    if result.exit_status != "SUCCESS":
        return False
    try:
        observed = json.loads(result.output_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return canonical_json(observed) == canonical_json(case.expected_output)


def run_local_solvability_proof(
    corpus: LearnableCorpusV2,
    workspace: Union[Path, str],
    *,
    splits: Sequence[str] = ("blind",),
) -> Dict[str, Any]:
    """Run a blind, public-only reference proof in the local test helper.

    This helper intentionally does not return hidden inputs or outputs.  It is
    a local functional proof and must not be interpreted as live Docker,
    OpenShell, Spark, or production evaluator acceptance.
    """

    selected_splits = tuple(splits)
    if not selected_splits or any(split not in SPLITS_V2 for split in selected_splits):
        raise ValueError("local solvability proof has an unknown split")
    sandbox = LocalTestSandbox(workspace)
    tasks = tuple(task for split in selected_splits for task in corpus.split(split))
    reference_failures: List[str] = []
    non_reference_semantic_passes: List[str] = []
    hidden_case_count = 0
    for task in tasks:
        package = validate_model_visible_package(corpus.model_visible_package(task.task_id))
        # The builder sees the closed public task record before the hidden-case
        # loop.  Its API cannot receive cases, expected values, or seed bytes.
        source = build_reference_source(package["task"])
        for case in task.hidden_cases:
            hidden_case_count += 1
            if not _semantic_local_result(sandbox, task, case, source, "reference"):
                reference_failures.append(case.case_id)
        starter = str(package["task"]["starter_source"]).encode("utf-8")
        first_case = task.hidden_cases[0]
        if _semantic_local_result(sandbox, task, first_case, starter, "non-reference"):
            non_reference_semantic_passes.append(first_case.case_id)
    report: Dict[str, Any] = {
        "schema_version": LEARNABLE_LOCAL_PROOF_VERSION,
        "evidence_scope": "local-test-helper-proof-not-production-acceptance",
        "production_acceptance": False,
        "runtime_backend": sandbox.backend_name,
        "splits": list(selected_splits),
        "task_count": len(tasks),
        "hidden_case_count": hidden_case_count,
        "builder_input": "closed-model-visible-task-record-only",
        "hidden_cases_passed_to_builder": False,
        "reference_all_hidden_cases_passed": not reference_failures,
        "reference_failure_count": len(reference_failures),
        "non_reference_public_only_candidate_rejected": not non_reference_semantic_passes,
        "non_reference_semantic_pass_count": len(non_reference_semantic_passes),
    }
    report["proof_digest"] = digest_for(report)
    return report


__all__ = [
    "FAMILY_SPECS_V2",
    "HIDDEN_CASES_PER_TASK",
    "INCOMPATIBLE_V1_VERSION",
    "LEARNABLE_DATA_VERSION",
    "LEARNABLE_LOCAL_PROOF_VERSION",
    "LEARNABLE_MODEL_PACKAGE_VERSION",
    "LEARNABLE_SEED_BYTES",
    "LEARNABLE_SEED_ENV",
    "LEARNABLE_SPEC_VERSION",
    "LEARNABLE_TASK_VERSION",
    "LearnableCase",
    "LearnableCorpusV2",
    "LearnableFamilySpec",
    "LearnableTask",
    "SPLITS_V2",
    "STAGE_A_TASK_IDS",
    "build_reference_source",
    "run_local_solvability_proof",
    "solve_public_spec",
    "validate_learnable_manifest",
    "validate_learnable_private_manifest",
    "validate_model_visible_package",
    "validate_public_task_record",
]
