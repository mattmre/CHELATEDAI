"""Inventory historical corpus exposure and enforce exact text disjointness."""

import argparse
import ast
import hashlib
import json
import re
import tokenize
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from . import _json_sha256, _read_json

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SPEC = "docs/research/bundle_registry_spec.json"

# Historical exposure survives a missing script, cache, or registry entry.
HISTORICAL_DATASETS = {
    "msmarco": "Wave 11 D10",
    "quora": "Waves 1–15 BEIR and Wave 11 D10",
    "fiqa": "Waves 1–15 BEIR and Wave 11 D10",
    "scidocs": "Waves 1–15 BEIR and Wave 11 D10",
    "arguana": "Waves 1–15 BEIR and Wave 11 D10",
    "scifact": "Waves 1–15 BEIR and Wave 11 D10",
    "nfcorpus": "Waves 1–15 BEIR and Wave 11 D10",
    "longmemeval": "Flagship 2 PM-1b and PM-4",
    "locomo": "Flagship 2 PM-1a and PM-4",
    "osv_cve_seeds": "Phase 2 D1 and Wave 11 D9",
}
_ALIAS_PATTERNS = {
    "msmarco": r"ms[-_. ]?marco",
    "quora": r"quora(?:[-_. ]?(?:qp|question[-_. ]pairs))?",
    "fiqa": r"fiqa(?:[-_. ]?2018)?",
    "scidocs": r"sci[-_. ]?docs",
    "arguana": r"argu[-_. ]?ana",
    "scifact": r"sci[-_. ]?fact",
    "nfcorpus": r"nf[-_. ]?corpus",
    "longmemeval": r"long[-_. ]?mem[-_. ]?eval(?:[-_. ](?:s|m)(?:[-_. ]cleaned)?|[-_. ]cleaned)?",
    "locomo": r"lo[-_. ]?co[-_. ]?mo(?:(?:[-_. ](?:mc)?\d+)|\d+)?",
    "osv_cve_seeds": r"osv[-_. ]cve[-_. ]seeds",
}
_ALIAS_RE = {
    key: re.compile(r"(?<![a-z0-9])(?:" + value + r")(?![a-z0-9])", re.IGNORECASE)
    for key, value in _ALIAS_PATTERNS.items()
}
_ID_FIELDS = frozenset({"dataset_id", "dataset_ids", "dataset", "datasets", "corpus_id", "corpus_ids"})
_DIGEST_FIELDS = {
    "doc_sha256": "documents",
    "document_sha256": "documents",
    "doc_fingerprints": "documents",
    "document_fingerprints": "documents",
    "query_sha256": "queries",
    "query_fingerprints": "queries",
}
_TEXT_FIELDS = {"doc_texts": "documents", "document_texts": "documents", "query_texts": "queries"}
_LOADERS = frozenset({"load_dataset", "hf_hub_download"})
_FILE_LOADERS = frozenset({"csv", "json", "parquet", "text", "arrow", "imagefolder", "audiofolder"})
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_DATASET_ID_RE = re.compile(r"[\w.-]+(?:/[\w.-]+)*\Z")


class BurnedCorpusError(ValueError):
    """Historical exposure or missing overlap evidence prevents admission."""


@dataclass(frozen=True)
class BurnLedger:
    """Source-attributed exposure index and exact historical text fingerprints."""

    datasets: tuple[tuple[str, tuple[str, ...]], ...]
    document_fingerprints: frozenset[str]
    query_fingerprints: frozenset[str]
    fingerprint_inventory_complete: bool = False

    def to_dict(self):
        return {
            "datasets": {
                name: {"status": "BURNED", "sources": list(sources)}
                for name, sources in self.datasets
            },
            "document_fingerprints": sorted(self.document_fingerprints),
            "query_fingerprints": sorted(self.query_fingerprints),
            "fingerprint_inventory_complete": self.fingerprint_inventory_complete,
        }

    @property
    def sha256(self):
        return _json_sha256(self.to_dict())


def normalize_dataset_id(dataset_id):
    """Canonicalize dataset names, Hugging Face namespaces and known variants."""
    if not isinstance(dataset_id, str):
        raise ValueError("dataset_id must be a string")
    value = unicodedata.normalize("NFKC", dataset_id).strip().casefold().rstrip("/")
    for prefix in ("https://huggingface.co/datasets/", "hf://datasets/"):
        if value.startswith(prefix):
            value = value.removeprefix(prefix)
            break
    value = re.sub(r"\s+", "-", value)
    if not _DATASET_ID_RE.fullmatch(value) or any(part in {".", ".."} for part in value.split("/")):
        raise ValueError(f"Invalid dataset ID: {dataset_id!r}")
    basename = value.rsplit("/", 1)[-1]
    basename = re.sub(r"[-_.]qrels$|\.(?:jsonl?|zip|parquet)$", "", basename)
    for canonical, pattern in _ALIAS_PATTERNS.items():
        if re.fullmatch(pattern, basename):
            return canonical
    return basename


def text_fingerprint(text):
    """Hash Unicode NFKC/casefolded text with collapsed Unicode whitespace."""
    if not isinstance(text, str):
        raise ValueError("Fingerprint input must be text")
    folded = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(unicodedata.normalize("NFKC", folded).split())
    if not normalized:
        raise ValueError("Fingerprint input cannot be empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _iter_values(value):
    if isinstance(value, Mapping):
        return value.values()
    if isinstance(value, str):
        return (value,)
    return iter(value)


def _digests(value):
    for item in _iter_values(value):
        if not isinstance(item, str) or not _SHA256_RE.fullmatch(item):
            raise ValueError(f"Expected a SHA-256 fingerprint, got {item!r}")
        yield item.lower()


def _protected(path):
    return any(
        "sealed_key" in part.casefold()
        or "blind-followup" in part.casefold()
        or part.casefold() == "pm4_human_audit_100_blind_sheet.csv"
        for part in path.parts
    )


def _check_source(path, root):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Historical source escapes repository: {path}")
    return not _protected(resolved.relative_to(root))


def _record_dataset(dataset_id, source, exposure):
    if isinstance(dataset_id, str) and dataset_id.strip().casefold() not in _FILE_LOADERS:
        try:
            dataset = normalize_dataset_id(dataset_id)
        except ValueError:
            return
        exposure[dataset].add(source)


def _read_field(key, value, source, exposure, fingerprints):
    key = key.casefold()
    if value is None:
        return
    if key in _ID_FIELDS:
        if isinstance(value, (str, list, tuple, set, Mapping)):
            for item in _iter_values(value):
                _record_dataset(item, source, exposure)
    elif key in _DIGEST_FIELDS:
        fingerprints[_DIGEST_FIELDS[key]].update(_digests(value))
    elif key in _TEXT_FIELDS:
        fingerprints[_TEXT_FIELDS[key]].update(map(text_fingerprint, _iter_values(value)))


def _registry_fields(node, source, exposure, fingerprints):
    if isinstance(node, dict):
        for key, value in node.items():
            _read_field(key, value, source, exposure, fingerprints)
            _registry_fields(value, source, exposure, fingerprints)
    elif isinstance(node, list):
        for value in node:
            _registry_fields(value, source, exposure, fingerprints)


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _scan_script(tree, source, exposure, fingerprints):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.casefold() in (
                    _ID_FIELDS | _DIGEST_FIELDS.keys() | _TEXT_FIELDS.keys()
                ):
                    value = _literal(node.value)
                    if value is not None:
                        _read_field(target.id, value, source, exposure, fingerprints)
        if not isinstance(node, ast.Call):
            continue
        func = getattr(node.func, "id", getattr(node.func, "attr", None))
        if func not in _LOADERS:
            continue
        options = {keyword.arg: _literal(keyword.value) for keyword in node.keywords}
        if func == "hf_hub_download" and options.get("repo_type") != "dataset":
            continue
        dataset = _literal(node.args[0]) if node.args else options.get("path", options.get("repo_id"))
        _record_dataset(dataset, source, exposure)


def _record_aliases(source, body, exposure):
    for name, pattern in _ALIAS_RE.items():
        if pattern.search(body) or pattern.search(source):
            exposure[name].add(source)


def _historical_inventories(fingerprints):
    if not isinstance(fingerprints, Mapping) or fingerprints.keys() != {"documents", "queries"}:
        raise ValueError("Historical fingerprint index requires documents and queries inventories")
    return set(_digests(fingerprints["documents"])), set(_digests(fingerprints["queries"]))


def build_burn_ledger(repo_root=REPO_ROOT, *, historical_fingerprints=None):
    """Read historical sources and optionally attach a supplied complete digest inventory.

    An inventory maps documents and queries to their normalized SHA-256
    fingerprints. Its supplier attests that it covers historical exposures.
    Script and registry references alone cannot establish inventory completeness.
    """
    root = Path(repo_root).resolve(strict=True)
    scripts = root / "scripts"
    if not _check_source(scripts, root) or not scripts.is_dir():
        raise ValueError("Historical scripts directory is unavailable")
    registry = root / REGISTRY_SPEC
    if not _check_source(registry, root):
        raise ValueError("Historical registry is unavailable")
    exposure = defaultdict(set)
    fingerprints = {"documents": set(), "queries": set()}
    for name, origin in HISTORICAL_DATASETS.items():
        exposure[name].add(f"historical:{origin}")

    spec = registry.read_text(encoding="utf-8-sig")
    _registry_fields(_read_json(spec), REGISTRY_SPEC, exposure, fingerprints)
    _record_aliases(REGISTRY_SPEC, spec, exposure)
    for path in sorted(scripts.glob("*.py")):
        relative = path.relative_to(root)
        if _protected(relative) or not _check_source(path, root):
            continue
        source = relative.as_posix()
        with tokenize.open(path) as stream:
            body = stream.read()
        _scan_script(ast.parse(body, filename=source), source, exposure, fingerprints)
        _record_aliases(source, body, exposure)
    documents, queries = fingerprints["documents"], fingerprints["queries"]
    complete = False
    if historical_fingerprints is not None:
        supplied_documents, supplied_queries = _historical_inventories(historical_fingerprints)
        complete = bool(supplied_documents and supplied_queries)
        documents.update(supplied_documents)
        queries.update(supplied_queries)
    datasets = tuple((name, tuple(sorted(sources))) for name, sources in sorted(exposure.items()))
    return BurnLedger(datasets, frozenset(documents), frozenset(queries), complete)


def assert_corpus_unburned(dataset_id, doc_texts=None, query_texts=None, historical_fingerprints=None, *, ledger=None):
    """Reject known IDs and digest collisions, including across text roles.

    For admission, supply a complete historical digest index as a mapping with
    explicit documents and queries inventories, or pass a ledger built with
    that inventory. An unlabelled digest iterable can establish a collision.
    """
    try:
        name = normalize_dataset_id(dataset_id)
        ledger = build_burn_ledger() if ledger is None else ledger
        sources = dict(ledger.datasets)
        if name in sources:
            raise BurnedCorpusError(f"Dataset {dataset_id!r} is BURNED as {name}: {', '.join(sources[name])}")

        historical = set()
        complete = ledger.fingerprint_inventory_complete
        if isinstance(historical_fingerprints, Mapping):
            supplied_documents, supplied_queries = _historical_inventories(historical_fingerprints)
            complete = complete or bool(supplied_documents and supplied_queries)
            historical = supplied_documents | supplied_queries
        elif historical_fingerprints is not None:
            historical.update(_digests(historical_fingerprints))

        for role, texts in (("document", doc_texts), ("query", query_texts)):
            if texts is None:
                continue
            if isinstance(texts, (str, bytes, Mapping)):
                raise ValueError(f"Candidate {role} texts must be supplied as an iterable of texts")
            index = 0
            for index, value in enumerate(texts, start=1):
                fingerprint = text_fingerprint(value)
                if (
                    fingerprint in historical
                    or fingerprint in ledger.document_fingerprints
                    or fingerprint in ledger.query_fingerprints
                ):
                    raise BurnedCorpusError(f"BURNED {role} #{index}: exact normalized SHA-256 collision {fingerprint}")
            if index == 0:
                raise ValueError(f"Candidate {role} inventory is empty")
        if doc_texts is None or query_texts is None or not complete:
            raise ValueError("Candidate document/query texts and a complete historical fingerprint index are required")
    except BurnedCorpusError:
        raise
    except (OSError, SyntaxError, TypeError, ValueError) as exc:
        raise BurnedCorpusError(f"Cannot establish corpus disjointness: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    options = parser.parse_args(argv)
    ledger = build_burn_ledger(options.repo_root)
    print(json.dumps({**ledger.to_dict(), "burn_ledger_sha256": ledger.sha256}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
