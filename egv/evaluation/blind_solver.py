"""Independent compiler for the model-visible EGV public expression package.

The compiler's only task-bearing input is a canonical JSON byte string.  It
validates that self-contained package without consulting corpus generation,
hidden cases, evaluator seeds, expected outputs, or the trusted oracle.  The
result is deterministic ``pure-return-v1`` Python source suitable for the
candidate sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..canonical import canonical_json, canonical_value, digest_bytes, digest_for
from .sandbox import validate_candidate_source_contract


BLIND_SOLVER_VERSION = "egv-independent-blind-solver-v1"

_PACKAGE_VERSION = "egv-learnable-model-package-v2"
_TASK_VERSION = "egv-learnable-task-v2"
_SPEC_VERSION = "egv-learnable-public-spec-v2"
_DSL_VERSION = "egv-public-expression-dsl-v1"
_INCOMPATIBLE_VERSION = "egv-evaluation-data-v1"
_MAX_PACKAGE_BYTES = 1_048_576
_MAX_EXPRESSION_DEPTH = 64
_MAX_EXPRESSION_NODES = 4096
_MAX_COLLECTION_ITEMS = 512
_MAX_PATH_PARTS = 32
_MAX_NAME_LENGTH = 256

_PACKAGE_FIELDS = frozenset({"schema_version", "task", "package_digest"})
_TASK_FIELDS = frozenset(
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
_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "expression_language",
        "family_id",
        "task_scope",
        "input_contract",
        "parameters",
        "payload_contract",
        "output_contract",
        "program",
    }
)
_EXPRESSION_FIELDS = {
    "literal": frozenset({"op", "value"}),
    "input": frozenset({"op", "path"}),
    "variable": frozenset({"op", "name", "path"}),
    "object": frozenset({"op", "fields"}),
    "add_integers": frozenset({"op", "items"}),
    "split_text": frozenset({"op", "text", "delimiter"}),
    "if_equal": frozenset({"op", "left", "right", "then", "else"}),
    "sort_objects": frozenset({"op", "items", "key"}),
    "map_list": frozenset({"op", "items", "variable", "body"}),
    "list_length": frozenset({"op", "items"}),
    "sum_integers": frozenset({"op", "items"}),
    "mapping_get": frozenset({"op", "mapping", "key", "default"}),
}

_STARTER_SOURCE = (
    "def main(value):\n"
    "    return {'case_nonce': value.get('case_nonce'), "
    "'task_scope': value.get('task_scope'), 'status': 'unimplemented'}\n"
)

_FAMILY_LAYOUT = {
    "PURE_FUNCTION": ("pure-function", {"train": 4, "dev": 2, "heldout": 20, "blind": 3}),
    "PARSER_EDGE": ("parser-edge", {"train": 4, "dev": 2, "heldout": 20, "blind": 3}),
    "STATE_TRANSITION": ("state-transition", {"train": 3, "dev": 2, "heldout": 20, "blind": 3}),
    "DATA_TRANSFORM": ("data-transform", {"train": 3, "dev": 2, "heldout": 20, "blind": 3}),
    "RESOURCE_BOUND": ("resource-bound", {"train": 3, "dev": 2, "heldout": 20, "blind": 3}),
    "DEPENDENCY_CONTRACT": ("dependency-contract", {"train": 3, "dev": 2, "heldout": 20, "blind": 3}),
}

_INPUT_CONTRACT = {
    "closed_fields": ["case_nonce", "payload", "task_scope"],
    "case_nonce": {"type": "text", "length": 20},
    "task_scope_must_match": True,
}


@dataclass(frozen=True)
class _ValueType:
    kind: str
    fields: Tuple[Tuple[str, "_ValueType"], ...] = ()
    item: Optional["_ValueType"] = None
    key: Optional["_ValueType"] = None
    value: Optional["_ValueType"] = None

    def field(self, name: str) -> "_ValueType":
        for field_name, field_type in self.fields:
            if field_name == name:
                return field_type
        raise ValueError("public expression path is unavailable")


_INTEGER = _ValueType("integer")
_NUMBER = _ValueType("number")
_TEXT = _ValueType("text")
_BOOLEAN = _ValueType("boolean")
_NULL = _ValueType("null")
_JSON = _ValueType("json")


@dataclass
class _ValidationState:
    nodes: int = 0

    def visit(self, depth: int) -> None:
        if depth > _MAX_EXPRESSION_DEPTH:
            raise ValueError("public expression exceeds the depth bound")
        self.nodes += 1
        if self.nodes > _MAX_EXPRESSION_NODES:
            raise ValueError("public expression exceeds the node bound")


@dataclass
class _CompileState:
    counter: int = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return "{}_{}".format(prefix, self.counter)


def _closed(record: Any, fields: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or set(record) != set(fields):
        raise ValueError("{} is not a closed record".format(label))
    return record


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_NAME_LENGTH:
        raise ValueError("{} is invalid".format(label))
    return value


def _parse_package(package_bytes: bytes) -> Mapping[str, Any]:
    if not isinstance(package_bytes, bytes):
        raise TypeError("model-visible package must be bytes")
    if not package_bytes or len(package_bytes) > _MAX_PACKAGE_BYTES:
        raise ValueError("model-visible package byte length is invalid")

    def reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("model-visible package contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        text = package_bytes.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant {}".format(value))
            ),
        )
        canonical = canonical_json(parsed).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("model-visible package is not valid canonical JSON") from exc
    if canonical != package_bytes:
        raise ValueError("model-visible package bytes are not canonical JSON")
    return parsed


def _digest_without(record: Mapping[str, Any], field: str) -> str:
    return digest_for({key: value for key, value in record.items() if key != field})


def _validate_task_identity(task: Mapping[str, Any]) -> None:
    family_id = task.get("family_id")
    split = task.get("split")
    ordinal = task.get("ordinal")
    if family_id not in _FAMILY_LAYOUT:
        raise ValueError("model-visible task has an unknown family")
    slug, split_limits = _FAMILY_LAYOUT[family_id]
    if split not in split_limits:
        raise ValueError("model-visible task has an unknown split")
    if not _is_integer(ordinal) or not 1 <= ordinal <= split_limits[split]:
        raise ValueError("model-visible task ordinal is outside the frozen allocation")
    expected_id = "egv-learnable-{}-{}-{:03d}-v2".format(slug, split, ordinal)
    if task.get("task_id") != expected_id:
        raise ValueError("model-visible task identity does not match its frozen allocation")


def _validate_family_contract(spec: Mapping[str, Any]) -> _ValueType:
    family_id = spec["family_id"]
    parameters = spec["parameters"]
    payload = spec["payload_contract"]
    output = spec["output_contract"]
    if not isinstance(parameters, Mapping):
        raise ValueError("public parameters must be an object")

    if family_id == "PURE_FUNCTION":
        _closed(parameters, {"bias"}, "PURE_FUNCTION parameters")
        if not _is_integer(parameters["bias"]):
            raise ValueError("PURE_FUNCTION bias must be an integer")
        expected_payload = {"type": "integer", "exclude_boolean": True}
        expected_output = ["case_nonce", "task_scope", "value"]
    elif family_id == "PARSER_EDGE":
        _closed(parameters, {"delimiter"}, "PARSER_EDGE parameters")
        if not isinstance(parameters["delimiter"], str) or not parameters["delimiter"]:
            raise ValueError("PARSER_EDGE delimiter must be non-empty text")
        expected_payload = {"type": "text"}
        expected_output = ["case_nonce", "task_scope", "tokens"]
    elif family_id == "STATE_TRANSITION":
        _closed(parameters, {"advance_by"}, "STATE_TRANSITION parameters")
        if not _is_integer(parameters["advance_by"]):
            raise ValueError("STATE_TRANSITION advance_by must be an integer")
        expected_payload = {
            "type": "closed_object",
            "fields": {
                "event": {"type": "enum", "values": ["reset", "advance"]},
                "state": {"type": "integer", "exclude_boolean": True},
            },
        }
        expected_output = ["case_nonce", "state", "task_scope"]
    elif family_id == "DATA_TRANSFORM":
        if parameters != {"sort_key": "key"}:
            raise ValueError("DATA_TRANSFORM parameters differ from the public contract")
        expected_payload = {
            "type": "non_empty_list",
            "items": {
                "type": "closed_object",
                "fields": {
                    "key": {"type": "text"},
                    "value": {"type": "integer", "exclude_boolean": True},
                },
            },
        }
        expected_output = ["case_nonce", "rows", "task_scope"]
    elif family_id == "RESOURCE_BOUND":
        _closed(parameters, {"max_items"}, "RESOURCE_BOUND parameters")
        max_items = parameters["max_items"]
        if not _is_integer(max_items) or max_items < 0:
            raise ValueError("RESOURCE_BOUND max_items must be a non-negative integer")
        expected_payload = {
            "type": "list",
            "max_items": max_items,
            "items": {"type": "integer", "exclude_boolean": True},
        }
        expected_output = ["case_nonce", "count", "sum", "task_scope"]
    elif family_id == "DEPENDENCY_CONTRACT":
        expected_default = "missing:" + spec["task_scope"]
        if parameters != {"default": expected_default}:
            raise ValueError("DEPENDENCY_CONTRACT default differs from the public contract")
        expected_payload = {
            "type": "closed_object",
            "fields": {
                "dependencies": {"type": "mapping", "key_type": "text", "value_type": "json"},
                "key": {"type": "text"},
            },
        }
        expected_output = ["case_nonce", "task_scope", "value"]
    else:  # guarded by task identity validation
        raise ValueError("public specification has an unknown family")

    if payload != expected_payload:
        raise ValueError("{} payload contract differs from the closed public contract".format(family_id))
    if output != expected_output:
        raise ValueError("{} output contract differs from the closed public contract".format(family_id))
    return _schema_type(payload)


def _schema_type(schema: Any) -> _ValueType:
    if not isinstance(schema, Mapping):
        raise ValueError("public value contract must be an object")
    kind = schema.get("type")
    if kind == "text" and set(schema) == {"type"}:
        return _TEXT
    if kind == "integer" and schema == {"type": "integer", "exclude_boolean": True}:
        return _INTEGER
    if kind == "enum" and set(schema) == {"type", "values"}:
        values = schema["values"]
        if (
            not isinstance(values, list)
            or not values
            or len(values) > _MAX_COLLECTION_ITEMS
            or any(not isinstance(value, str) for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("public enum contract is invalid")
        return _TEXT
    if kind == "closed_object" and set(schema) == {"type", "fields"}:
        fields = schema["fields"]
        if not isinstance(fields, Mapping) or not fields or len(fields) > _MAX_COLLECTION_ITEMS:
            raise ValueError("public closed-object contract is invalid")
        typed_fields = []
        for name, child in fields.items():
            typed_fields.append((_valid_name(name, "public contract field name"), _schema_type(child)))
        return _ValueType("object", fields=tuple(typed_fields))
    if kind in {"list", "non_empty_list"} and set(schema) in (
        {"type", "items"},
        {"type", "items", "max_items"},
    ):
        if "max_items" in schema and (not _is_integer(schema["max_items"]) or schema["max_items"] < 0):
            raise ValueError("public list bound is invalid")
        return _ValueType("list", item=_schema_type(schema["items"]))
    if kind == "mapping" and set(schema) == {"type", "key_type", "value_type"}:
        if schema["key_type"] != "text" or schema["value_type"] != "json":
            raise ValueError("public mapping contract is unsupported")
        return _ValueType("mapping", key=_TEXT, value=_JSON)
    raise ValueError("public value contract is unknown or not closed")


def _literal_type(value: Any) -> _ValueType:
    canonical = canonical_value(value)
    if canonical is None:
        return _NULL
    if isinstance(canonical, bool):
        return _BOOLEAN
    if isinstance(canonical, int):
        return _INTEGER
    if isinstance(canonical, float):
        return _NUMBER
    if isinstance(canonical, str):
        return _TEXT
    if isinstance(canonical, list):
        if not canonical:
            return _ValueType("list", item=_JSON)
        item_types = [_literal_type(item) for item in canonical]
        item_type = item_types[0] if all(item == item_types[0] for item in item_types) else _JSON
        return _ValueType("list", item=item_type)
    if isinstance(canonical, Mapping):
        return _ValueType(
            "object",
            fields=tuple((name, _literal_type(item)) for name, item in canonical.items()),
        )
    raise ValueError("public literal is not canonical JSON")


def _path_type(root: _ValueType, path: Any) -> _ValueType:
    if (
        not isinstance(path, list)
        or len(path) > _MAX_PATH_PARTS
        or any(not isinstance(part, str) or not part or len(part) > _MAX_NAME_LENGTH for part in path)
    ):
        raise ValueError("public expression path is invalid")
    current = root
    for part in path:
        if current.kind != "object":
            raise ValueError("public expression path traverses a non-object value")
        current = current.field(part)
    return current


def _compatible(left: _ValueType, right: _ValueType) -> bool:
    return left == right or left.kind == "json" or right.kind == "json"


def _validate_expression(
    node: Any,
    root: _ValueType,
    variables: Mapping[str, _ValueType],
    state: _ValidationState,
    depth: int = 0,
) -> _ValueType:
    state.visit(depth)
    if not isinstance(node, Mapping):
        raise ValueError("public expression node must be an object")
    op = node.get("op")
    if op not in _EXPRESSION_FIELDS:
        raise ValueError("public expression uses an unknown operator")
    if set(node) != set(_EXPRESSION_FIELDS[op]):
        raise ValueError("public expression node is not closed")

    def child(value: Any, child_variables: Optional[Mapping[str, _ValueType]] = None) -> _ValueType:
        return _validate_expression(value, root, child_variables or variables, state, depth + 1)

    if op == "literal":
        return _literal_type(node["value"])
    if op == "input":
        return _path_type(root, node["path"])
    if op == "variable":
        name = _valid_name(node["name"], "public expression variable name")
        if name not in variables:
            raise ValueError("public expression variable is unavailable")
        return _path_type(variables[name], node["path"])
    if op == "object":
        fields = node["fields"]
        if not isinstance(fields, list) or not fields or len(fields) > _MAX_COLLECTION_ITEMS:
            raise ValueError("public expression object fields are invalid")
        typed_fields = []
        names = set()
        for field in fields:
            _closed(field, {"name", "value"}, "public expression object field")
            name = _valid_name(field["name"], "public expression object field name")
            if name in names:
                raise ValueError("public expression object field is duplicated")
            names.add(name)
            typed_fields.append((name, child(field["value"])))
        return _ValueType("object", fields=tuple(typed_fields))
    if op == "add_integers":
        items = node["items"]
        if not isinstance(items, list) or not items or len(items) > _MAX_COLLECTION_ITEMS:
            raise ValueError("add_integers item list is invalid")
        if any(child(item) != _INTEGER for item in items):
            raise ValueError("add_integers operands must be integers")
        return _INTEGER
    if op == "split_text":
        if child(node["text"]) != _TEXT or child(node["delimiter"]) != _TEXT:
            raise ValueError("split_text operands must be text")
        delimiter = node["delimiter"]
        if delimiter.get("op") == "literal" and delimiter.get("value") == "":
            raise ValueError("split_text delimiter must be non-empty")
        return _ValueType("list", item=_TEXT)
    if op == "if_equal":
        left = child(node["left"])
        right = child(node["right"])
        if not _compatible(left, right):
            raise ValueError("if_equal operands have incompatible public types")
        then_type = child(node["then"])
        else_type = child(node["else"])
        if not _compatible(then_type, else_type):
            raise ValueError("if_equal branches have incompatible public types")
        return then_type if then_type.kind != "json" else else_type
    if op == "sort_objects":
        items_type = child(node["items"])
        key = _valid_name(node["key"], "sort_objects key")
        if items_type.kind != "list" or items_type.item is None or items_type.item.kind != "object":
            raise ValueError("sort_objects requires a list of closed objects")
        key_type = items_type.item.field(key)
        if key_type.kind not in {"integer", "number", "text"}:
            raise ValueError("sort_objects key has an unsupported public type")
        return items_type
    if op == "map_list":
        items_type = child(node["items"])
        variable = _valid_name(node["variable"], "map_list variable")
        if items_type.kind != "list" or items_type.item is None:
            raise ValueError("map_list requires a public list")
        body_type = child(node["body"], {**variables, variable: items_type.item})
        return _ValueType("list", item=body_type)
    if op == "list_length":
        if child(node["items"]).kind != "list":
            raise ValueError("list_length requires a public list")
        return _INTEGER
    if op == "sum_integers":
        items_type = child(node["items"])
        if items_type.kind != "list" or items_type.item != _INTEGER:
            raise ValueError("sum_integers requires an integer list")
        return _INTEGER
    if op == "mapping_get":
        mapping_type = child(node["mapping"])
        key_type = child(node["key"])
        default_type = child(node["default"])
        if mapping_type.kind != "mapping" or mapping_type.key is None or mapping_type.value is None:
            raise ValueError("mapping_get requires a public mapping")
        if not _compatible(mapping_type.key, key_type):
            raise ValueError("mapping_get key differs from the public mapping contract")
        if not _compatible(mapping_type.value, default_type):
            raise ValueError("mapping_get default differs from the public mapping contract")
        return mapping_type.value
    raise AssertionError("validated public expression operator was not handled")


def _validate_spec(spec: Any, task: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = _closed(spec, _SPEC_FIELDS, "public expression specification")
    if spec.get("schema_version") != _SPEC_VERSION:
        raise ValueError("unsupported public expression specification")
    if spec.get("expression_language") != _DSL_VERSION:
        raise ValueError("unsupported public expression language")
    if spec.get("family_id") != task["family_id"]:
        raise ValueError("public specification family differs from the task")
    if spec.get("task_scope") != task["task_id"]:
        raise ValueError("public specification scope differs from the task")
    if spec.get("input_contract") != _INPUT_CONTRACT:
        raise ValueError("public input contract differs from the closed contract")
    payload_type = _validate_family_contract(spec)
    root_type = _ValueType(
        "object",
        fields=(("case_nonce", _TEXT), ("payload", payload_type), ("task_scope", _TEXT)),
    )
    output_type = _validate_expression(spec["program"], root_type, {}, _ValidationState())
    if output_type.kind != "object":
        raise ValueError("public program output must be a closed object")
    if [name for name, _ in output_type.fields] != spec["output_contract"]:
        raise ValueError("public program fields differ from the output contract")
    return spec


def validate_model_visible_package_bytes(package_bytes: bytes) -> Dict[str, Any]:
    """Validate and detach one canonical model-visible package byte string."""

    package = _closed(_parse_package(package_bytes), _PACKAGE_FIELDS, "model-visible package")
    if package.get("schema_version") != _PACKAGE_VERSION:
        raise ValueError("unsupported model-visible package schema")
    if package.get("package_digest") != _digest_without(package, "package_digest"):
        raise ValueError("model-visible package digest mismatch")

    task = _closed(package.get("task"), _TASK_FIELDS, "model-visible task")
    if task.get("schema_version") != _TASK_VERSION:
        raise ValueError("unsupported model-visible task schema")
    if task.get("incompatible_with") != [_INCOMPATIBLE_VERSION]:
        raise ValueError("model-visible task does not declare the required incompatibility")
    _validate_task_identity(task)
    spec = _validate_spec(task.get("public_spec"), task)
    if task.get("public_spec_digest") != digest_for(spec):
        raise ValueError("public specification digest mismatch")
    if task.get("starter_source") != _STARTER_SOURCE:
        raise ValueError("starter source differs from the frozen public package")
    if task.get("starter_source_digest") != digest_bytes(_STARTER_SOURCE.encode("utf-8")):
        raise ValueError("starter source digest mismatch")
    if task.get("record_digest") != _digest_without(task, "record_digest"):
        raise ValueError("model-visible task record digest mismatch")
    return canonical_value(package)


def _compile_path(root: str, path: Sequence[str]) -> str:
    return root + "".join("[{!r}]".format(part) for part in path)


def _compile_expression(
    node: Mapping[str, Any],
    variables: Mapping[str, str],
    state: _CompileState,
) -> str:
    op = node["op"]
    if op == "literal":
        return repr(canonical_value(node["value"]))
    if op == "input":
        return _compile_path("value", node["path"])
    if op == "variable":
        return _compile_path(variables[node["name"]], node["path"])
    if op == "object":
        return "{" + ", ".join(
            "{!r}: {}".format(field["name"], _compile_expression(field["value"], variables, state))
            for field in node["fields"]
        ) + "}"
    if op == "add_integers":
        compiled = [_compile_expression(item, variables, state) for item in node["items"]]
        return "sum(({},))".format(", ".join(compiled))
    if op == "split_text":
        return "{}.split({})".format(
            _compile_expression(node["text"], variables, state),
            _compile_expression(node["delimiter"], variables, state),
        )
    if op == "if_equal":
        return "({} if {} == {} else {})".format(
            _compile_expression(node["then"], variables, state),
            _compile_expression(node["left"], variables, state),
            _compile_expression(node["right"], variables, state),
            _compile_expression(node["else"], variables, state),
        )
    if op == "sort_objects":
        items = _compile_expression(node["items"], variables, state)
        item_name = state.name("_blind_sort_item")
        return "sorted({}, key=lambda {}: {}[{!r}])".format(items, item_name, item_name, node["key"])
    if op == "map_list":
        items = _compile_expression(node["items"], variables, state)
        item_name = state.name("_blind_map_item")
        body = _compile_expression(node["body"], {**variables, node["variable"]: item_name}, state)
        return "[{} for {} in {}]".format(body, item_name, items)
    if op == "list_length":
        return "len({})".format(_compile_expression(node["items"], variables, state))
    if op == "sum_integers":
        return "sum({})".format(_compile_expression(node["items"], variables, state))
    if op == "mapping_get":
        return "{}.get({}, {})".format(
            _compile_expression(node["mapping"], variables, state),
            _compile_expression(node["key"], variables, state),
            _compile_expression(node["default"], variables, state),
        )
    raise AssertionError("validated public expression operator was not compiled")


def compile_blind_source(package_bytes: bytes) -> bytes:
    """Compile candidate source from canonical model-visible package bytes only."""

    package = validate_model_visible_package_bytes(package_bytes)
    spec = package["task"]["public_spec"]
    expression = _compile_expression(spec["program"], {}, _CompileState())
    source = (
        "def main(value):\n"
        "    if value.get('task_scope') != {!r}:\n"
        "        return {{'error': 'scope-mismatch'}}\n"
        "    _blind_result = {}\n"
        "    return _blind_result\n"
    ).format(spec["task_scope"], expression).encode("utf-8")
    violation = validate_candidate_source_contract(source)
    if violation is not None:
        raise ValueError("blind source violates pure-return-v1: {}".format(violation))
    return source


__all__ = [
    "BLIND_SOLVER_VERSION",
    "compile_blind_source",
    "validate_model_visible_package_bytes",
]
