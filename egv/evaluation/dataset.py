"""Deterministic, split-disjoint evaluation micro-repositories.

The public template ID is only an address.  Generator seeds, hidden inputs,
hidden rules, identifier vocabularies, and golden repairs are derived from an
evaluator-only seed file.  The seed is never accepted on a command line and is
never written to trainer or public artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from ..canonical import canonical_json, digest_bytes, digest_for
from .errors import EvaluationError


SPLITS = ("train", "dev", "heldout")
DATA_MANIFEST_VERSION = "egv-evaluation-data-v1"
EVALUATOR_SEED_ENV = "EGV_EVALUATOR_SEED_FILE"
EVALUATOR_SEED_BYTES = 32


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    train: int
    dev: int
    heldout: int
    problem: str

    def count(self, split: str) -> int:
        if split == "train":
            return self.train
        if split == "dev":
            return self.dev
        if split == "heldout":
            return self.heldout
        raise ValueError("unknown evaluation split")


FAMILY_SPECS = (
    FamilySpec("PURE_FUNCTION", 4, 1, 1, "repair a deterministic pure-function result"),
    FamilySpec("PARSER_EDGE", 4, 1, 1, "repair tokenization at a grammar edge"),
    FamilySpec("STATE_TRANSITION", 3, 2, 1, "repair a finite-state transition invariant"),
    FamilySpec("DATA_TRANSFORM", 3, 2, 1, "repair a schema-preserving transformation"),
    FamilySpec("RESOURCE_BOUND", 3, 1, 2, "meet a frozen resource ceiling"),
    FamilySpec("DEPENDENCY_CONTRACT", 3, 1, 2, "repair a generated local dependency API"),
)
FAMILY_BY_ID = {spec.family_id: spec for spec in FAMILY_SPECS}


def _template_id(family_id: str, split: str, ordinal: int) -> str:
    return "egv-{}-{}-{}-v1".format(family_id.lower(), split, ordinal)


# This is the public, closed identifier vocabulary used by publishable-root
# scanners.  It contains addresses only; expected values and repair bodies
# remain evaluator-seed derived and private.
PUBLIC_HELDOUT_TEMPLATE_IDS = frozenset(
    _template_id(spec.family_id, "heldout", ordinal)
    for spec in FAMILY_SPECS
    for ordinal in range(1, spec.heldout + 1)
)


def _read_evaluator_seed(secret_seed_file: Optional[Union[Path, str]]) -> bytes:
    configured = secret_seed_file or os.environ.get(EVALUATOR_SEED_ENV)
    if not configured:
        raise EvaluationError(
            "Evaluation corpus generation requires an evaluator-only seed file; "
            "set EGV_EVALUATOR_SEED_FILE in the evaluator process or pass secret_seed_file"
        )
    path = Path(configured)
    try:
        secret = path.read_bytes()
    except OSError as exc:
        raise EvaluationError("evaluator seed file is unavailable") from exc
    if len(secret) != EVALUATOR_SEED_BYTES:
        raise EvaluationError("evaluator seed file must contain exactly 32 bytes")
    return secret


def _derive(secret: bytes, label: str) -> str:
    return hmac.new(secret, label.encode("utf-8"), hashlib.sha256).hexdigest()


def _task_material(secret: bytes, family_id: str, split: str, ordinal: int) -> Dict[str, str]:
    scope = "egv-evaluation-v2|{}|{}|{}".format(family_id, split, ordinal)
    return {
        "generator_seed": _derive(secret, scope + "|generator-seed"),
        "public_token": _derive(secret, scope + "|public-identifier")[:16],
        "hidden_token": _derive(secret, scope + "|hidden-fixture")[:24],
        "fixture_token": _derive(secret, scope + "|fixture-rule")[:24],
        # This value is an evaluator-only oracle input.  It is deliberately
        # independent from the public identifier and from the opaque input so
        # a public-only repair cannot infer the accepted result.
        "oracle_token": _derive(secret, scope + "|evaluator-oracle")[:32],
    }


def _source_files(family: FamilySpec, public_token: str) -> Tuple[Tuple[str, bytes], ...]:
    name = "solve_{}".format(public_token)
    if family.family_id == "PURE_FUNCTION":
        body = "def {}(value):\n    return 'public:' + str(value)\n".format(name)
    elif family.family_id == "PARSER_EDGE":
        body = "def {}(text):\n    return text.split(' ')\n".format(name)
    elif family.family_id == "STATE_TRANSITION":
        body = "def {}(state):\n    return {{'state': state, 'transition': 'PUBLIC'}}\n".format(name)
    elif family.family_id == "DATA_TRANSFORM":
        body = "def {}(rows):\n    return [row['public'] for row in rows]\n".format(name)
    elif family.family_id == "RESOURCE_BOUND":
        body = "def {}(items):\n    return len(items)\n".format(name)
    else:
        body = "def {}(value, dependency):\n    return dependency.get(value)\n".format(name)
    if family.family_id == "DEPENDENCY_CONTRACT":
        main = "\ndef main(value):\n    return {}(value, {{}})\n".format(name)
    else:
        main = "\ndef main(value):\n    return {}(value)\n".format(name)
    readme = (
        "# egv-evaluation-microrepo\n\nFamily: {}\n\n"
        "Repair the bounded function while preserving the public API.\n"
        "The evaluator supplies one opaque input and reports only a closed diagnostic.\n"
    ).format(family.family_id)
    return (("README.md", readme.encode("utf-8")), ("src/task.py", (body + main).encode("utf-8")))


def _hidden_spec(family: FamilySpec, material: Mapping[str, str]) -> Dict[str, Any]:
    hidden = material["hidden_token"]
    oracle = material["oracle_token"]
    if family.family_id == "PURE_FUNCTION":
        input_value = "input-{}".format(hidden)
        expected = "answer-{}".format(oracle)
        golden = "return the evaluator-only oracle answer"
    elif family.family_id == "PARSER_EDGE":
        # The opaque input carries only independent hidden fixture material;
        # the accepted middle token is a separate evaluator-only oracle.  In
        # particular, the input must not contain the oracle as a delimiter,
        # because a generic parser could otherwise recover it.
        separator = material["fixture_token"]
        input_value = "alpha{}{}{}omega".format(separator, hidden, separator)
        expected = ["alpha", oracle, "omega"]
        golden = "return tokens split on the evaluator-only separator"
    elif family.family_id == "STATE_TRANSITION":
        input_value = "state-{}".format(hidden)
        expected = {"state": input_value, "transition": "transition-{}".format(oracle)}
        golden = "return the evaluator-only transition oracle"
    elif family.family_id == "DATA_TRANSFORM":
        input_value = [{"private": "{}-first".format(hidden)}, {"private": "{}-second".format(hidden)}]
        expected = ["{}-first".format(oracle), "{}-second".format(oracle)]
        golden = "read the evaluator-only corrected schema values"
    elif family.family_id == "RESOURCE_BOUND":
        input_value = ["resource-{}".format(hidden)]
        expected = "bounded-{}".format(oracle)
        golden = "return the evaluator-only bounded oracle under the frozen memory ceiling"
        resource_limit = "64m"
    else:
        input_value = "key-{}".format(hidden)
        expected = "value-{}".format(oracle)
        golden = "return the evaluator-only dependency value"
        resource_limit = None
    result = {
        "hidden_rule_id": "rule-hidden-{}".format(material["fixture_token"]),
        "input": input_value,
        "expected_output": expected,
        "golden_patch": "{}::{}".format(golden, material["fixture_token"]),
        "oracle_token": oracle,
    }
    if family.family_id == "RESOURCE_BOUND":
        result["resource_limit"] = resource_limit
    return result


def _corrected_source(
    family: FamilySpec,
    split: str,
    public_token: str,
    hidden_spec: Mapping[str, Any],
    hidden_token: str,
    fixture_token: str,
    generator_seed: str,
    oracle_token: str,
) -> bytes:
    name = "solve_{}".format(public_token)
    shape_token = generator_seed[2:14]
    shape_name = "PRIVATE_SHAPE_{}".format(shape_token)
    shape_value = generator_seed[14:30]
    if split not in SPLITS:
        raise ValueError("unknown repair split")
    # These are intentionally different implementation families, not merely
    # different HMAC labels.  The public validator records and checks the
    # algorithm identity, while the source below changes control flow and
    # data-shaping operations for every split.
    if family.family_id == "PURE_FUNCTION":
        if split == "train":
            body = (
                "{} = {!r}\n"
                "def {}(value):\n"
                "    return 'answer-' + {!r} if {} else 'invalid'\n"
            ).format(shape_name, shape_value, name, oracle_token, shape_name)
        elif split == "dev":
            body = (
                "{} = {!r}\n"
                "def _answer_{}():\n"
                "    return ''.join((\"answer-\", {!r}))\n"
                "def {}(value):\n"
                "    return _answer_{}() if bool({}) else 'invalid'\n"
            ).format(shape_name, shape_value, shape_name, oracle_token, name, shape_name, shape_name)
        else:
            body = (
                "{} = {!r}\n"
                "def {}(value):\n"
                "    parts = ['answer', {!r}]\n"
                "    return '-'.join(parts) if all(({}, parts[1])) else 'invalid'\n"
            ).format(shape_name, shape_value, name, oracle_token, shape_name)
    elif family.family_id == "PARSER_EDGE":
        separator = fixture_token
        if split == "train":
            body = (
                "{} = {!r}\n"
                "PRIVATE_SEPARATOR_{} = {!r}\n"
                "PRIVATE_ORACLE_{} = {!r}\n"
                "def {}(text):\n"
                "    parts = [item for item in text.split(PRIVATE_SEPARATOR_{}) if item]\n"
                "    return ['alpha', PRIVATE_ORACLE_{}, 'omega'] if len(parts) == 3 and {} else []\n"
            ).format(shape_name, shape_value, fixture_token, separator, fixture_token, oracle_token, name, fixture_token, fixture_token, shape_name)
        elif split == "dev":
            body = (
                "{} = {!r}\n"
                "PRIVATE_SEPARATOR_{} = {!r}\n"
                "PRIVATE_ORACLE_{} = {!r}\n"
                "def {}(text):\n"
                "    pieces = text.split(PRIVATE_SEPARATOR_{})\n"
                "    tokens = []\n"
                "    for piece in pieces:\n"
                "        if piece:\n"
                "            tokens.append(piece)\n"
                "    return ['alpha', PRIVATE_ORACLE_{}, 'omega'] if len(tokens) == 3 and {} else []\n"
            ).format(shape_name, shape_value, fixture_token, separator, fixture_token, oracle_token, name, fixture_token, fixture_token, shape_name)
        else:
            body = (
                "{} = {!r}\n"
                "PRIVATE_SEPARATOR_{} = {!r}\n"
                "PRIVATE_ORACLE_{} = {!r}\n"
                "def {}(text):\n"
                "    parts = list(filter(lambda item: item != '', text.split(PRIVATE_SEPARATOR_{})))\n"
                "    return ['alpha', PRIVATE_ORACLE_{}, 'omega'] if len(parts) == 3 and {} else []\n"
            ).format(shape_name, shape_value, fixture_token, separator, fixture_token, oracle_token, name, fixture_token, fixture_token, shape_name)
    elif family.family_id == "STATE_TRANSITION":
        transition = "transition-{}".format(oracle_token)
        if split == "train":
            body = (
                "{} = {!r}\n"
                "PRIVATE_TRANSITION_{} = {!r}\n"
                "def {}(state):\n"
                "    return {{'state': state, 'transition': PRIVATE_TRANSITION_{}}} if {} else {{}}\n"
            ).format(shape_name, shape_value, fixture_token, transition, name, fixture_token, shape_name)
        elif split == "dev":
            body = (
                "{} = {!r}\n"
                "def {}(state):\n"
                "    values = [('state', state), ('transition', 'transition-' + {!r})]\n"
                "    return dict(values) if {} else {{}}\n"
            ).format(shape_name, shape_value, name, oracle_token, shape_name)
        else:
            body = (
                "{} = {!r}\n"
                "def {}(state):\n"
                "    result = {{}}\n"
                "    result.update(state=state)\n"
                "    result.update(transition='transition-{{}}'.format({!r}))\n"
                "    return result if {} else {{}}\n"
            ).format(shape_name, shape_value, name, oracle_token, shape_name)
    elif family.family_id == "DATA_TRANSFORM":
        first = "{}-first".format(oracle_token)
        second = "{}-second".format(oracle_token)
        if split == "train":
            body = (
                "{} = {!r}\n"
                "PRIVATE_FIELD_{} = 'private'\n"
                "PRIVATE_FIRST_{} = {!r}\n"
                "PRIVATE_SECOND_{} = {!r}\n"
                "def {}(rows):\n"
                "    return [PRIVATE_FIRST_{}, PRIVATE_SECOND_{}] if len(rows) == 2 and {} else []\n"
            ).format(shape_name, shape_value, fixture_token, fixture_token, first, fixture_token, second, name, fixture_token, fixture_token, shape_name)
        elif split == "dev":
            body = (
                "{} = {!r}\n"
                "PRIVATE_VALUES_{} = ({!r}, {!r})\n"
                "def {}(rows):\n"
                "    values = []\n"
                "    for index, row in enumerate(rows):\n"
                "        if 'private' not in row:\n"
                "            return []\n"
                "        values.append(PRIVATE_VALUES_{}[index])\n"
                "    return values if len(values) == 2 and {} else []\n"
            ).format(shape_name, shape_value, fixture_token, first, second, name, fixture_token, shape_name)
        else:
            body = (
                "{} = {!r}\n"
                "PRIVATE_VALUES_{} = [{!r}, {!r}]\n"
                "def {}(rows):\n"
                "    return list(map(lambda pair: PRIVATE_VALUES_{}[pair[0]], enumerate(rows))) if len(rows) == 2 and {} else []\n"
            ).format(shape_name, shape_value, fixture_token, first, second, name, fixture_token, shape_name)
    elif family.family_id == "RESOURCE_BOUND":
        bounded = "bounded-{}".format(oracle_token)
        if split == "train":
            body = (
                "{} = {!r}\n"
                "PRIVATE_RESOURCE_{} = {!r}\n"
                "def {}(items):\n"
                "    return {!r} if items and items[0].startswith('resource-') and {} else 'bounded-invalid'\n"
            ).format(shape_name, shape_value, fixture_token, "resource-" + hidden_token, name, bounded, shape_name)
        elif split == "dev":
            body = (
                "{} = {!r}\n"
                "def {}(items):\n"
                "    first = items[0] if items else None\n"
                "    return {!r} if first is not None and first.startswith('resource-') and {} else 'bounded-invalid'\n"
            ).format(shape_name, shape_value, name, bounded, shape_name)
        else:
            body = (
                "{} = {!r}\n"
                "def {}(items):\n"
                "    valid = bool(items)\n"
                "    for item in items[:1]:\n"
                "        valid = valid and item.startswith('resource-')\n"
                "    return {!r} if valid and {} else 'bounded-invalid'\n"
            ).format(shape_name, shape_value, name, bounded, shape_name)
    else:
        key = hidden_spec["input"]
        value = hidden_spec["expected_output"]
        if split == "train":
            body = (
                "{} = {!r}\n"
                "PRIVATE_KEY_{} = {!r}\n"
                "PRIVATE_VALUE_{} = {!r}\n"
                "def {}(value, dependency):\n"
                "    return dependency[value] if value == PRIVATE_KEY_{} and {} else None\n"
            ).format(shape_name, shape_value, fixture_token, key, fixture_token, value, name, fixture_token, shape_name)
        elif split == "dev":
            body = (
                "{} = {!r}\n"
                "def {}(value, dependency):\n"
                "    return dict(dependency).get(value) if {} else None\n"
            ).format(shape_name, shape_value, name, shape_name)
        else:
            body = (
                "{} = {!r}\n"
                "def {}(value, dependency):\n"
                "    for key, item in dependency.items():\n"
                "        if key == value and {}:\n"
                "            return item\n"
                "    return None\n"
            ).format(shape_name, shape_value, name, shape_name)
    # Every corrected body carries a secret-derived fixture tag.  This is a
    # deliberate content-binding marker, not a public rule: without the
    # evaluator seed an exact repair body cannot be reconstructed.
    body = (
        "PRIVATE_FIXTURE_{} = {!r}\n"
        "PRIVATE_HIDDEN_{} = {!r}\n"
    ).format(fixture_token, fixture_token, hidden_token, hidden_token) + body
    if family.family_id == "DEPENDENCY_CONTRACT":
        expected = hidden_spec["expected_output"]
        main = (
            "\nPRIVATE_DEPENDENCY_FIXTURE_{} = {!r}\n"
            "def main(value):\n"
            "    return {}(value, {{{!r}: {!r}}})\n"
        ).format(fixture_token, fixture_token, name, hidden_spec["input"], expected)
    else:
        main = "\ndef main(value):\n    return {}(value)\n".format(name)
    return (body + main).encode("utf-8")


@dataclass(frozen=True)
class MicroRepo:
    template_id: str
    family_id: str
    split: str
    ordinal: int
    source_files: Tuple[Tuple[str, bytes], ...]
    hidden_spec: Mapping[str, Any]
    source_digest: str
    hidden_spec_digest: str
    public_rule_id: str
    public_locus: str
    private_material: Mapping[str, Any]
    corrected_source: bytes

    @property
    def evaluator_input(self) -> Any:
        return self.hidden_spec["input"]

    @property
    def expected_output(self) -> Any:
        return self.hidden_spec["expected_output"]

    def public_manifest_record(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "family_id": self.family_id,
            "split": self.split,
            "ordinal": self.ordinal,
            "source_digest": self.source_digest,
            "public_rule_id": self.public_rule_id,
            "public_locus": self.public_locus,
        }

    def private_manifest_record(self) -> Dict[str, Any]:
        return {
            **self.public_manifest_record(),
            "hidden_spec_digest": self.hidden_spec_digest,
            "corrected_source_digest": digest_bytes(self.corrected_source),
        }

    def private_hidden_record(self, *, seed_digest: str) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "seed_file_digest": seed_digest,
            **dict(self.hidden_spec),
        }


def _build_repo(secret: bytes, family: FamilySpec, split: str, ordinal: int) -> MicroRepo:
    template_id = _template_id(family.family_id, split, ordinal)
    material = _task_material(secret, family.family_id, split, ordinal)
    source_files = _source_files(family, material["public_token"])
    hidden_spec = _hidden_spec(family, material)
    source_digest = digest_for({path: data.decode("utf-8") for path, data in source_files})
    hidden_digest = digest_for(hidden_spec)
    name = "solve_{}".format(material["public_token"])
    corrected_source = _corrected_source(
        family,
        split,
        material["public_token"],
        hidden_spec,
        material["hidden_token"],
        material["fixture_token"],
        material["generator_seed"],
        material["oracle_token"],
    )
    private_material = {
        "generator_seed": material["generator_seed"],
        "oracle_token": material["oracle_token"],
        "repair_algorithm_id": "repair-{}-{}-v2".format(split, family.family_id),
        "repair_algorithm_skeleton": digest_for("{}|{}|v2".format(split, family.family_id)),
        "correction_shape_token": material["generator_seed"][2:14],
        "correction_shape_value": material["generator_seed"][14:30],
        "fixture_id": "fixture-{}".format(material["fixture_token"]),
        "identifier_vocabulary": [
            name,
            "PRIVATE_SHAPE_{}".format(material["generator_seed"][2:14]),
            "rule-public-{}".format(material["public_token"]),
            "rule-hidden-{}".format(material["fixture_token"]),
            material["hidden_token"],
        ],
        "corrected_source_secret_tokens": [
            material["public_token"],
            material["hidden_token"],
            material["fixture_token"],
            material["generator_seed"][2:14],
            material["generator_seed"][14:30],
            material["oracle_token"],
        ],
        "hidden_rule_id": hidden_spec["hidden_rule_id"],
        "golden_patch": hidden_spec["golden_patch"],
        "corrected_source_digest": digest_bytes(corrected_source),
    }
    return MicroRepo(
        template_id=template_id,
        family_id=family.family_id,
        split=split,
        ordinal=ordinal,
        source_files=source_files,
        hidden_spec=hidden_spec,
        source_digest=source_digest,
        hidden_spec_digest=hidden_digest,
        public_rule_id="rule-public-{}".format(material["public_token"]),
        public_locus="module:{}".format(name),
        private_material=private_material,
        corrected_source=corrected_source,
    )


class EvaluationCorpus:
    """Frozen corpus generator and split-isolation validator."""

    def __init__(self, repositories: Sequence[MicroRepo], secret_seed: bytes) -> None:
        self.repositories = tuple(repositories)
        self._by_id = {repo.template_id: repo for repo in self.repositories}
        self._secret_seed = bytes(secret_seed)
        self.validate()

    @classmethod
    def generate(cls, *, secret_seed_file: Optional[Union[Path, str]] = None) -> "EvaluationCorpus":
        secret = _read_evaluator_seed(secret_seed_file)
        repositories: List[MicroRepo] = []
        for family in FAMILY_SPECS:
            for split in SPLITS:
                for ordinal in range(1, family.count(split) + 1):
                    repositories.append(_build_repo(secret, family, split, ordinal))
        return cls(repositories, secret)

    @property
    def secret_seed_digest(self) -> str:
        return digest_for(self._secret_seed)

    def private_seed_bytes(self) -> bytes:
        """Return the seed only to the evaluator artifact writer."""

        return bytes(self._secret_seed)

    def validate(self) -> None:
        if len(self._secret_seed) != EVALUATOR_SEED_BYTES:
            raise ValueError("evaluation corpus has an invalid evaluator seed")
        expected_ids = []
        for family in FAMILY_SPECS:
            for split in SPLITS:
                for ordinal in range(1, family.count(split) + 1):
                    expected_ids.append(_template_id(family.family_id, split, ordinal))
        actual_ids = [repo.template_id for repo in self.repositories]
        if actual_ids != expected_ids:
            raise ValueError("evaluation corpus IDs, ordering, or counts differ from the frozen contract")
        if len(set(actual_ids)) != len(actual_ids):
            raise ValueError("evaluation corpus contains duplicate template IDs")
        for repo in self.repositories:
            if repo.split not in SPLITS or repo.family_id not in FAMILY_BY_ID:
                raise ValueError("evaluation corpus contains an unknown family or split")
            if repo.source_digest != digest_for({path: data.decode("utf-8") for path, data in repo.source_files}):
                raise ValueError("micro-repository source digest mismatch")
            if repo.hidden_spec_digest != digest_for(repo.hidden_spec):
                raise ValueError("micro-repository hidden-spec digest mismatch")
            if repo.private_material["corrected_source_digest"] != digest_bytes(repo.corrected_source):
                raise ValueError("micro-repository corrected-source digest mismatch")
            corrected_text = repo.corrected_source.decode("utf-8")
            for token in repo.private_material["corrected_source_secret_tokens"]:
                if str(token) not in corrected_text:
                    raise ValueError("corrected source is not bound to evaluator-only generator material")
            public_text = b"\n".join(data for _path, data in repo.source_files).decode("utf-8")
            if repo.hidden_spec["hidden_rule_id"] in public_text:
                raise ValueError("hidden rule leaked into public source")
            if str(repo.hidden_spec["golden_patch"]) in public_text:
                raise ValueError("golden patch leaked into public source")
            if canonical_json(repo.expected_output) in public_text:
                raise ValueError("hidden expected output is reconstructable from public source")
        self.validate_split_disjointness()

    def validate_split_disjointness(self) -> Dict[str, Any]:
        """Prove that secret-derived generator material is pairwise split-disjoint."""

        fields = (
            "generator_seed",
            "correction_shape_token",
            "correction_shape_value",
            "fixture_id",
            "hidden_rule_id",
            "identifier_vocabulary",
            "corrected_source_secret_tokens",
            "golden_patch",
            "corrected_source_digest",
            "oracle_token",
            "repair_algorithm_id",
            "repair_algorithm_skeleton",
        )
        records: Dict[str, Dict[str, Set[str]]] = {}
        corrected_source_records: Dict[str, Set[bytes]] = {}
        task_source_records: Dict[str, Set[bytes]] = {}
        for split in SPLITS:
            records[split] = {}
            corrected_source_records[split] = {repo.corrected_source for repo in self.split(split)}
            task_source_records[split] = {
                data for repo in self.split(split) for path, data in repo.source_files if path == "src/task.py"
            }
            for field in fields:
                values: Set[str] = set()
                for repo in self.split(split):
                    value = repo.private_material[field]
                    if isinstance(value, list):
                        values.update(str(item) for item in value)
                    else:
                        values.add(str(value))
                records[split][field] = values
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                for field in fields:
                    overlap = records[left][field] & records[right][field]
                    if overlap:
                        raise ValueError("split-disjoint material overlaps for {} and {}: {}".format(left, right, field))
                if corrected_source_records[left] & corrected_source_records[right]:
                    raise ValueError("corrected source bodies overlap across {} and {}".format(left, right))
                if task_source_records[left] & task_source_records[right]:
                    raise ValueError("public task source bodies overlap across {} and {}".format(left, right))
        return {
            "checked_fields": list(fields),
            "checked_content": ["corrected_source_bytes", "task_source_bytes"],
            "split_pairs": ["{}:{}".format(left, right) for left_index, left in enumerate(SPLITS) for right in SPLITS[left_index + 1 :]],
            "pairwise_disjoint": True,
        }

    def get(self, template_id: str) -> MicroRepo:
        try:
            return self._by_id[template_id]
        except KeyError as exc:
            raise KeyError("unknown evaluation template ID") from exc

    def split(self, split: str) -> Tuple[MicroRepo, ...]:
        if split not in SPLITS:
            raise ValueError("unknown evaluation split")
        return tuple(repo for repo in self.repositories if repo.split == split)

    def manifest(self) -> Dict[str, Any]:
        records = [repo.private_manifest_record() for repo in self.repositories]
        return {
            "schema_version": DATA_MANIFEST_VERSION,
            "task_count": len(records),
            "split_counts": {split: len(self.split(split)) for split in SPLITS},
            "family_counts": {
                family.family_id: {split: family.count(split) for split in SPLITS} for family in FAMILY_SPECS
            },
            "seed_file_digest": self.secret_seed_digest,
            "tasks": records,
        }

    def manifest_digest(self) -> str:
        return digest_for(self.manifest())

    def public_summary(self) -> Dict[str, Any]:
        return {
            "schema_version": DATA_MANIFEST_VERSION,
            "task_count": len(self.repositories),
            "split_counts": {split: len(self.split(split)) for split in SPLITS},
            "family_counts": {
                family.family_id: {split: family.count(split) for split in SPLITS} for family in FAMILY_SPECS
            },
            "manifest_digest": self.manifest_digest(),
        }

    def trainer_repositories(self) -> Tuple[MicroRepo, ...]:
        return tuple(repo for repo in self.repositories if repo.split in {"train", "dev"})

    def hidden_repositories(self) -> Tuple[MicroRepo, ...]:
        return self.split("heldout")

    def hidden_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-evaluation-hidden-v1",
            "task_count": len(self.hidden_repositories()),
            "seed_file_digest": self.secret_seed_digest,
            "tasks": [repo.private_manifest_record() for repo in self.hidden_repositories()],
        }

    def correct_candidate_source(self, template_id: str) -> bytes:
        """Return the private generated ``src/task.py`` candidate source."""

        return bytes(self.get(template_id).corrected_source)


def validate_data_manifest(manifest: Mapping[str, Any], corpus: Optional[EvaluationCorpus] = None) -> Dict[str, Any]:
    """Validate a frozen manifest against the exact corpus contract."""

    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != DATA_MANIFEST_VERSION:
        raise ValueError("unsupported evaluation data manifest")
    expected_fields = {"schema_version", "task_count", "split_counts", "family_counts", "seed_file_digest", "tasks"}
    if set(manifest) != expected_fields:
        raise ValueError("evaluation data manifest is not closed")
    expected = corpus or EvaluationCorpus.generate()
    expected_manifest = expected.manifest()
    if manifest != expected_manifest:
        raise ValueError("data manifest differs from deterministic evaluator regeneration")
    if manifest.get("task_count") != 36:
        raise ValueError("data manifest must enumerate exactly 36 repositories")
    return dict(manifest)


__all__ = [
    "DATA_MANIFEST_VERSION",
    "EVALUATOR_SEED_BYTES",
    "EVALUATOR_SEED_ENV",
    "EvaluationCorpus",
    "FAMILY_SPECS",
    "FamilySpec",
    "MicroRepo",
    "PUBLIC_HELDOUT_TEMPLATE_IDS",
    "SPLITS",
    "validate_data_manifest",
]
