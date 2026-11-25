"""Validate and compile drop-in registry, route, and verifier fragments."""

import argparse
import ast
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath

from . import _json_sha256, _read_json
from .claim_registry import BUNDLE_IDS, STATUSES

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_DIRS = {
    "registry": "docs/research/registry.d",
    "routes": "demo_workbench/routes.d",
    "verifiers": "scripts/verifiers.d",
}
_JSON_NAME = re.compile(r"(?P<id>[0-9]{2})-[a-z0-9]+(?:-[a-z0-9]+)*\.json\Z")
_VERIFIER_NAME = re.compile(r"(?P<id>[0-9]{2})_verifier\.py\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_HTTP_METHOD = re.compile(r"[!#$%&'*+.^_a-zA-Z0-9|~-]+\Z")
_ROUTE_PATH = re.compile(r"/[a-zA-Z0-9._~!$&'()*+,;=:@/-]*\Z")
_PATH_FIELDS = {"dir", "directory", "manifest", "receipt", "artifact", "artifacts", "ancillary_json_hashes"}
_METADATA = {"SPINE_DIGEST", "BUNDLE_ID", "DEPENDS_ON"}


class FragmentCompileError(ValueError):
    """A fragment cannot be safely included in the compiled spine."""


def _protected_path(parts):
    return any(
        "sealed_key" in part.casefold()
        or "blind-followup" in part.casefold()
        or part.casefold() == "pm4_human_audit_100_blind_sheet.csv"
        for part in parts
    )


def _path_parts(value):
    return value.replace("\\", "/").split("/")


def _checked_path(root, value):
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        raise FragmentCompileError(f"Invalid fragment path: {value!r}")
    parts = _path_parts(value)
    if any(part and (part in {".", ".."} or part.endswith((" ", "."))) for part in parts):
        raise FragmentCompileError(f"Path traversal or ambiguous path: {value!r}")
    if _protected_path(parts):
        raise FragmentCompileError(f"Protected fragment path: {value!r}")
    win = PureWindowsPath(value)
    local = Path(value.replace("\\", "/"))
    # A foreign drive, UNC share or drive-relative path cannot be resolved as a repo path.
    if (win.drive or win.root) and not local.is_absolute():
        raise FragmentCompileError(f"Path escapes repository: {value!r}")
    if ":" in value[len(win.drive):]:
        raise FragmentCompileError(f"Invalid fragment path: {value!r}")
    path = (root / local).resolve()
    if not path.is_relative_to(root):
        raise FragmentCompileError(f"Path escapes repository: {value!r}")
    if _protected_path(path.relative_to(root).parts):
        raise FragmentCompileError(f"Protected fragment path: {value!r}")
    return path


def _validate_reference_paths(root, values, path_context=False):
    if isinstance(values, dict):
        for key, value in values.items():
            _validate_reference_paths(root, key)
            name = key.casefold()
            refers_to_path = name in _PATH_FIELDS or name.endswith(
                ("_path", "_paths", "_file", "_files", "_dir", "_dirs", "_manifest", "_receipt")
            ) or name in {"path", "paths", "file", "files"}
            if path_context and name == "sha256":
                continue
            _validate_reference_paths(root, value, path_context or refers_to_path)
    elif isinstance(values, list):
        for value in values:
            _validate_reference_paths(root, value, path_context)
    elif isinstance(values, str):
        parts = _path_parts(values)
        looks_like_path = bool(
            values.startswith(("/", "\\", "../", "..\\")) or re.match(r"^[A-Za-z]:", values)
            or (".." in parts and not any(char.isspace() for char in values))
        )
        if path_context or looks_like_path:
            _checked_path(root, values)


def _fragment_name(relative):
    if not isinstance(relative, str):
        raise FragmentCompileError(f"Invalid fragment filename: {relative!r}")
    parent, _, filename = relative.rpartition("/")
    for kind, directory in FRAGMENT_DIRS.items():
        if parent != directory:
            continue
        pattern = _VERIFIER_NAME if kind == "verifiers" else _JSON_NAME
        if match := pattern.fullmatch(filename):
            bundle_id = int(match["id"])
            if bundle_id in BUNDLE_IDS:
                return kind, bundle_id
        break
    raise FragmentCompileError(f"Invalid fragment filename or out-of-range bundle ID: {relative!r}")


def _verifier_metadata(root, source, filename):
    # Compile for syntax and scope checks. Verifier code is never executed here.
    tree = ast.parse(source, filename=filename)
    compile(tree, filename, "exec", dont_inherit=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            _validate_reference_paths(root, node.value)
    declarations = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in _METADATA:
                continue
            if target.id in declarations or statement.value is None:
                raise FragmentCompileError(f"Invalid verifier declaration {target.id}: {filename}")
            try:
                declarations[target.id] = ast.literal_eval(statement.value)
            except (ValueError, TypeError) as exc:
                raise FragmentCompileError(f"Verifier declaration {target.id} must be literal: {filename}") from exc
    return {
        "id": declarations.get("BUNDLE_ID"),
        "spine_digest": declarations.get("SPINE_DIGEST"),
        "depends_on": declarations.get("DEPENDS_ON", ()),
    }


def _dependencies(fragment, filename):
    deps = fragment.get("depends_on", [])
    if not isinstance(deps, (list, tuple)):
        raise FragmentCompileError(f"Invalid depends_on in {filename}")
    if any(type(dep) is not int or dep not in BUNDLE_IDS for dep in deps) or len(deps) != len(set(deps)):
        raise FragmentCompileError(f"Invalid or duplicate dependency IDs in {filename}")
    return set(deps)


def _load_fragment(root, kind, bundle_id, filename, content, expected_spine_digest):
    try:
        if kind == "verifiers":
            fragment = _verifier_metadata(root, content, filename)
        else:
            fragment = _read_json(content)
    except (ValueError, SyntaxError) as exc:
        raise FragmentCompileError(f"Invalid fragment {filename}: {exc}") from exc
    if not isinstance(fragment, dict):
        raise FragmentCompileError(f"Expected an object in {filename}")
    declared_id = fragment.get("id", bundle_id)
    if kind == "verifiers" and declared_id is None:
        declared_id = bundle_id
    if type(declared_id) is not int or declared_id != bundle_id:
        raise FragmentCompileError(f"Bundle ID does not match filename: {filename}")
    if fragment.get("spine_digest") != expected_spine_digest:
        raise FragmentCompileError(f"Mismatched spine_digest: {filename}")
    return fragment


def _collect_routes(root, bundle_id, fragment, routes):
    entries = fragment.get("routes")
    if not isinstance(entries, list) or not entries:
        raise FragmentCompileError(f"Bundle {bundle_id} requires a non-empty routes list")
    _validate_reference_paths(root, {key: value for key, value in fragment.items() if key != "routes"})
    for entry in entries:
        if not isinstance(entry, dict):
            raise FragmentCompileError(f"Bundle {bundle_id} route must be an object")
        method, path = entry.get("method"), entry.get("path")
        if not isinstance(method, str) or not _HTTP_METHOD.fullmatch(method):
            raise FragmentCompileError(f"Invalid route method in bundle {bundle_id}: {method!r}")
        if not isinstance(path, str) or not _ROUTE_PATH.fullmatch(path) or "//" in path or any(
            segment in {".", ".."} for segment in path.split("/")
        ):
            raise FragmentCompileError(f"Invalid or noncanonical route path in bundle {bundle_id}: {path!r}")
        key = method.upper(), path
        if key in routes:
            raise FragmentCompileError(f"Duplicate HTTP route: {key[0]} {key[1]}")
        if "bundle_id" in entry and (type(entry["bundle_id"]) is not int or entry["bundle_id"] != bundle_id):
            raise FragmentCompileError(f"Route bundle_id does not match fragment: {bundle_id}")
        _validate_reference_paths(root, {field: value for field, value in entry.items() if field != "path"})
        routes[key] = {**entry, "method": key[0], "bundle_id": bundle_id}


def _dependency_order(graph):
    order, visiting, visited = [], set(), set()

    def visit(bundle_id):
        if bundle_id in visiting:
            raise FragmentCompileError(f"Dependency cycle involving bundle {bundle_id}")
        if bundle_id in visited:
            return
        if bundle_id not in graph:
            raise FragmentCompileError(f"Missing registry fragment for dependency {bundle_id}")
        visiting.add(bundle_id)
        for dependency in sorted(graph[bundle_id]):
            visit(dependency)
        visiting.remove(bundle_id)
        visited.add(bundle_id)
        order.append(bundle_id)

    for bundle_id in sorted(graph):
        visit(bundle_id)
    return order


def compile_fragments(repo_root, expected_spine_digest, sources):
    """Compile path-keyed UTF-8 fragment bytes to registries, routes and verifier descriptors.

    JSON fragments declare spine_digest and may declare id and depends_on. The
    registry fragment additionally declares status. A route fragment has routes
    with method/path entries. Python verifiers declare literal SPINE_DIGEST
    and may declare BUNDLE_ID and DEPENDS_ON. Only registry fragments create a
    bundle. Route and verifier fragments amend their corresponding bundle.
    """
    if not isinstance(expected_spine_digest, str) or not _SHA256.fullmatch(expected_spine_digest):
        raise FragmentCompileError("Expected the pinned spine SHA-256 as 64 lowercase hex digits")
    if not isinstance(sources, Mapping) or any(not isinstance(item, str) for item in sources):
        raise FragmentCompileError("Fragment sources must be a mapping keyed by relative filename")
    root = Path(repo_root).resolve(strict=True)
    registry, routes, verifiers, graph = {}, {}, {}, {}
    seen = set()
    for filename, raw in sorted(sources.items()):
        kind, bundle_id = _fragment_name(filename)
        _checked_path(root, filename)
        if (kind, bundle_id) in seen:
            raise FragmentCompileError(f"Duplicate {kind} fragment for bundle {bundle_id}")
        seen.add((kind, bundle_id))
        if not isinstance(raw, bytes):
            raise FragmentCompileError(f"Fragment content must be bytes: {filename}")
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise FragmentCompileError(f"Fragment is not UTF-8: {filename}") from exc
        fragment = _load_fragment(root, kind, bundle_id, filename, content, expected_spine_digest)
        graph.setdefault(bundle_id, set()).update(_dependencies(fragment, filename))
        if kind == "registry":
            status = fragment.get("status")
            if not isinstance(status, str) or status not in STATUSES - {"reserved-pending"}:
                raise FragmentCompileError(f"Invalid published registry status: {filename}")
            _validate_reference_paths(root, fragment)
            registry[bundle_id] = {**fragment, "id": bundle_id}
        elif kind == "routes":
            _collect_routes(root, bundle_id, fragment, routes)
        else:
            verifiers[bundle_id] = {
                "bundle_id": bundle_id,
                "path": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
    if orphaned := graph.keys() - registry.keys():
        raise FragmentCompileError(f"Missing registry fragments for bundles: {sorted(orphaned)}")
    order = _dependency_order(graph)
    compiled = {
        "spine_digest": expected_spine_digest,
        "registry": [registry[bundle_id] for bundle_id in sorted(registry)],
        "routes": [routes[key] for key in sorted(routes)],
        "verifiers": [verifiers[bundle_id] for bundle_id in order if bundle_id in verifiers],
        "dependency_order": order,
    }
    return {**compiled, "compiled_sha256": _json_sha256(compiled)}


def compile_repo_fragments(repo_root, expected_spine_digest):
    """Discover the three drop-in directories and read fragments relative to repo root."""
    root = Path(repo_root).resolve(strict=True)
    sources = {}
    for kind, relative in FRAGMENT_DIRS.items():
        directory = _checked_path(root, relative)
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise FragmentCompileError(f"Fragment directory is not a directory: {relative}")
        extension = "*.py" if kind == "verifiers" else "*.json"
        for file in directory.glob(extension):
            filename = f"{relative}/{file.name}"
            _fragment_name(filename)
            path = _checked_path(root, filename)
            sources[filename] = path.read_bytes()
    return compile_fragments(root, expected_spine_digest, sources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--spine-digest", required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_repo_fragments(args.repo_root, args.spine_digest)
    except (OSError, FragmentCompileError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
