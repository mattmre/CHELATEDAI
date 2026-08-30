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


def _public_spec(task_id: str, family_id: str) -> Dict[str, Any]:
    material = _public_material(task_id)
    common = {
        "schema_version": LEARNABLE_SPEC_VERSION,
        "family_id": family_id,
        "task_scope": task_id,
        "input_contract": {
            "closed_fields": ["case_nonce", "payload", "task_scope"],
            "task_scope_must_match": True,
        },
    }
    if family_id == "PURE_FUNCTION":
        common.update(
            {
                "operation": "double-plus-bias",
                "parameters": {"bias": 5 + material[0] % 19},
                "output_contract": ["case_nonce", "task_scope", "value"],
            }
        )
    elif family_id == "PARSER_EDGE":
        delimiters = ("::", "||", "~|~")
        common.update(
            {
                "operation": "split-preserve-empty",
                "parameters": {"delimiter": delimiters[material[0] % len(delimiters)]},
                "output_contract": ["case_nonce", "task_scope", "tokens"],
            }
        )
    elif family_id == "STATE_TRANSITION":
        common.update(
            {
                "operation": "reset-or-advance",
                "parameters": {"advance_by": 1 + material[0] % 7},
                "output_contract": ["case_nonce", "state", "task_scope"],
            }
        )
    elif family_id == "DATA_TRANSFORM":
        common.update(
            {
                "operation": "sort-and-double-values",
                "parameters": {"sort_key": "key"},
                "output_contract": ["case_nonce", "rows", "task_scope"],
            }
        )
    elif family_id == "RESOURCE_BOUND":
        common.update(
            {
                "operation": "bounded-sum-and-count",
                "parameters": {"max_items": 16 + material[0] % 17},
                "output_contract": ["case_nonce", "count", "sum", "task_scope"],
            }
        )
    elif family_id == "DEPENDENCY_CONTRACT":
        common.update(
            {
                "operation": "mapping-get-with-public-default",
                "parameters": {"default": "missing:" + task_id},
                "output_contract": ["case_nonce", "task_scope", "value"],
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
    return runtime_input


def solve_public_spec(public_spec: Mapping[str, Any], runtime_input: Any) -> Any:
    """Return the unique answer using public spec and runtime input only."""

    if not isinstance(public_spec, Mapping):
        raise ValueError("public spec must be an object")
    family_id = public_spec.get("family_id")
    task_scope = public_spec.get("task_scope")
    if not isinstance(task_scope, str) or public_spec != _public_spec(task_scope, str(family_id)):
        raise ValueError("public spec differs from the frozen functional contract")
    value = _validate_runtime_input(public_spec, runtime_input)
    payload = value["payload"]
    nonce = value["case_nonce"]
    if family_id == "PURE_FUNCTION":
        if not isinstance(payload, int) or isinstance(payload, bool):
            raise ValueError("PURE_FUNCTION payload must be an integer")
        result = payload + payload + public_spec["parameters"]["bias"]
        return {"case_nonce": nonce, "task_scope": task_scope, "value": result}
    if family_id == "PARSER_EDGE":
        if not isinstance(payload, str):
            raise ValueError("PARSER_EDGE payload must be text")
        tokens = payload.split(public_spec["parameters"]["delimiter"])
        return {"case_nonce": nonce, "task_scope": task_scope, "tokens": tokens}
    if family_id == "STATE_TRANSITION":
        if not isinstance(payload, Mapping) or set(payload) != {"event", "state"}:
            raise ValueError("STATE_TRANSITION payload is invalid")
        if not isinstance(payload["state"], int) or isinstance(payload["state"], bool):
            raise ValueError("STATE_TRANSITION state must be an integer")
        if payload["event"] == "reset":
            state = 0
        elif payload["event"] == "advance":
            state = payload["state"] + public_spec["parameters"]["advance_by"]
        else:
            raise ValueError("STATE_TRANSITION event is invalid")
        return {"case_nonce": nonce, "state": state, "task_scope": task_scope}
    if family_id == "DATA_TRANSFORM":
        if not isinstance(payload, list) or not payload:
            raise ValueError("DATA_TRANSFORM payload must be a non-empty list")
        if any(not isinstance(row, Mapping) or set(row) != {"key", "value"} for row in payload):
            raise ValueError("DATA_TRANSFORM row is invalid")
        rows = [
            {"key": row["key"], "value": row["value"] + row["value"]}
            for row in sorted(payload, key=lambda row: row["key"])
        ]
        return {"case_nonce": nonce, "rows": rows, "task_scope": task_scope}
    if family_id == "RESOURCE_BOUND":
        if not isinstance(payload, list) or len(payload) > public_spec["parameters"]["max_items"]:
            raise ValueError("RESOURCE_BOUND payload exceeds the public bound")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in payload):
            raise ValueError("RESOURCE_BOUND items must be integers")
        return {"case_nonce": nonce, "count": len(payload), "sum": sum(payload), "task_scope": task_scope}
    if family_id == "DEPENDENCY_CONTRACT":
        if not isinstance(payload, Mapping) or set(payload) != {"dependencies", "key"}:
            raise ValueError("DEPENDENCY_CONTRACT payload is invalid")
        dependencies = payload["dependencies"]
        if not isinstance(dependencies, Mapping) or not isinstance(payload["key"], str):
            raise ValueError("DEPENDENCY_CONTRACT mapping or key is invalid")
        result = dependencies.get(payload["key"], public_spec["parameters"]["default"])
        return {"case_nonce": nonce, "task_scope": task_scope, "value": result}
    raise ValueError("unknown learnable-v2 family")


def _reference_source_for_spec(public_spec: Mapping[str, Any]) -> bytes:
    family_id = public_spec["family_id"]
    scope = public_spec["task_scope"]
    prefix = (
        "def main(value):\n"
        "    if value.get('task_scope') != {!r}:\n"
        "        return {{'error': 'wrong-task'}}\n"
    ).format(scope)
    if family_id == "PURE_FUNCTION":
        body = (
            "    payload = value['payload']\n"
            "    return {{'case_nonce': value['case_nonce'], 'task_scope': {!r}, "
            "'value': payload + payload + {!r}}}\n"
        ).format(scope, public_spec["parameters"]["bias"])
    elif family_id == "PARSER_EDGE":
        body = (
            "    return {{'case_nonce': value['case_nonce'], 'task_scope': {!r}, "
            "'tokens': value['payload'].split({!r})}}\n"
        ).format(scope, public_spec["parameters"]["delimiter"])
    elif family_id == "STATE_TRANSITION":
        body = (
            "    payload = value['payload']\n"
            "    if payload['event'] == 'reset':\n"
            "        state = 0\n"
            "    else:\n"
            "        state = payload['state'] + {!r}\n"
            "    return {{'case_nonce': value['case_nonce'], 'state': state, 'task_scope': {!r}}}\n"
        ).format(public_spec["parameters"]["advance_by"], scope)
    elif family_id == "DATA_TRANSFORM":
        body = (
            "    rows = [{{'key': row['key'], 'value': row['value'] + row['value']}} "
            "for row in sorted(value['payload'], key=lambda row: row['key'])]\n"
            "    return {{'case_nonce': value['case_nonce'], 'rows': rows, 'task_scope': {!r}}}\n"
        ).format(scope)
    elif family_id == "RESOURCE_BOUND":
        body = (
            "    payload = value['payload']\n"
            "    return {{'case_nonce': value['case_nonce'], 'count': len(payload), "
            "'sum': sum(payload), 'task_scope': {!r}}}\n"
        ).format(scope)
    elif family_id == "DEPENDENCY_CONTRACT":
        body = (
            "    payload = value['payload']\n"
            "    result = payload['dependencies'].get(payload['key'], {!r})\n"
            "    return {{'case_nonce': value['case_nonce'], 'task_scope': {!r}, 'value': result}}\n"
        ).format(public_spec["parameters"]["default"], scope)
    else:
        raise ValueError("unknown learnable-v2 family")
    source = (prefix + body).encode("utf-8")
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
    if not isinstance(starter, str) or record.get("starter_source_digest") != digest_bytes(starter.encode("utf-8")):
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
                if case.input_digest != digest_for(case.input_value):
                    raise ValueError("learnable-v2 hidden input digest mismatch")
                expected_case_id = "case_{}".format(
                    digest_for({"task_id": task.task_id, "ordinal": case.ordinal, "input_digest": case.input_digest})
                )
                if case.case_id != expected_case_id:
                    raise ValueError("learnable-v2 hidden case ID mismatch")
                public_expected = solve_public_spec(task.public_record["public_spec"], case.input_value)
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
