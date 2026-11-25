"""Bundle #49 cold-start gate for the chelated-kernel wheel.

The 36 sessions are a deterministic local smoke. This process does not call a model.
live_model_calls stays 0. Pass rule: at least 30/36 sessions, at least 8/12 in each
family, zero private imports, 100 percent unsupported-capability canary accuracy,
and zero cross-tenant leaks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from importlib.abc import MetaPathFinder
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

CANARY_A = "PLANTED-CANARY-TENANT-A-9c1e"
CANARY_B = "PLANTED-CANARY-TENANT-B-4d2a"
MAGIC = np.float64(777777.25)
LOC_LIMIT = 2500
FAMILIES = ("fable-5.2", "astra-6.2", "public-reference")
VERTICALS = (
    "enterprise_search_migration",
    "sec_item_402_restatement",
    "osv_csaf_vex_conflict",
    "stream_suppression_holdback",
)
SEEDS = (0, 1, 2)
FAMILY_DIMS = {
    "fable-5.2": (8, 8),
    "astra-6.2": (8, 6),
    "public-reference": (6, 8),
}
UNSUPPORTED = (
    "bare_metal_p4_tcam_asic_dataplane",
    "physical_weight_erasure_unlearning",
    "hdd_servo_mcu_firmware",
    "cross_tenant_raw_join",
)
ALLOWED_LICENSES = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "Zlib",
        "CC0-1.0",
        "ISC",
        "Unlicense",
        "Python-2.0",
    }
)
_EXTRA_DIRS = ("egv", "demo_workbench", "computational_storage_poc", "formal", "scripts", "tests", "rlm_reference")
_GUARD_ATTR = "_chelated_kernel_repo_guard"


def synthetic_pair_anchors(
    seed: int,
    old_dim: int,
    new_dim: int,
    n_anchors: int = 64,
    n_docs: int = 5,
) -> tuple[list[tuple[list[float], list[float]]], list[list[float]], list[list[float]]]:
    """Exact linear paired anchors. The fit prefix is centered so the map is recoverable."""

    rng = np.random.default_rng(seed)
    if new_dim >= old_dim:
        mapping = np.linalg.qr(rng.normal(size=(new_dim, old_dim)))[0]
    else:
        mapping = np.linalg.qr(rng.normal(size=(old_dim, new_dim)))[0].T
    old = rng.normal(size=(n_anchors, old_dim))
    fit_count = 32
    if n_anchors < fit_count or fit_count <= old_dim:
        raise ValueError("synthetic fit prefix must be taller than old_dim")
    fit = old[:fit_count]
    fit = fit - fit.mean(axis=0, keepdims=True)
    chol = np.linalg.cholesky(fit.T @ fit)
    # Zero-mean and isotropic fit rows make the cross-covariance factor the
    # generating map, including when new_dim is smaller than old_dim.
    old[:fit_count] = fit @ np.linalg.inv(chol).T
    new = old @ mapping.T
    docs = np.eye(old_dim, dtype=np.float64)[:n_docs] * 5.0
    queries = docs @ mapping.T
    anchors = [
        (old[index].astype(float).tolist(), new[index].astype(float).tolist())
        for index in range(n_anchors)
    ]
    doc_rows = [docs[index].astype(float).tolist() for index in range(n_docs)]
    query_rows = [queries[index].astype(float).tolist() for index in range(n_docs)]
    return anchors, doc_rows, query_rows


def _package_dir(repo_root: Path) -> Path:
    return repo_root / "packages" / "chelated_kernel"


def _kernel_path(repo_root: Path) -> str:
    return str(_package_dir(repo_root))


def _ensure_kernel_path(repo_root: Path) -> None:
    root = _kernel_path(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    repo = str(repo_root)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def count_loc(repo_root: Path) -> int:
    total = 0
    for path in sorted((_package_dir(repo_root) / "chelated_kernel").glob("*.py")):
        total += len(path.read_text(encoding="utf-8").splitlines())
    return total


def _repo_internal_modules(repo_root: Path) -> frozenset[str]:
    names = {path.stem for path in repo_root.glob("*.py")}
    names.update(_EXTRA_DIRS)
    return frozenset(names)


class _RepoImportGuard(MetaPathFinder):
    """Block private kernel imports, and block repo modules when the caller is the package."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.blocked = _repo_internal_modules(repo_root)
        self.package_marker = "/packages/chelated_kernel/"

    def find_spec(self, fullname: str, path: object, target: object = None) -> None:  # type: ignore[override]
        private = fullname.startswith("chelated_kernel._") or (
            fullname.startswith("chelated_kernel.") and any(part.startswith("_") for part in fullname.split("."))
        )
        if private:
            raise ImportError(f"private import blocked: {fullname}")
        root = fullname.split(".")[0]
        if root not in self.blocked:
            return None
        frame = sys._getframe(1)
        for _step in range(40):
            if frame is None:
                break
            filename = str(frame.f_code.co_filename).replace("\\", "/")
            if self.package_marker in filename:
                raise ImportError(f"repo-internal import blocked: {fullname}")
            frame = frame.f_back
        return None


def install_import_guard(repo_root: Path) -> _RepoImportGuard:
    existing = getattr(install_import_guard, _GUARD_ATTR, None)
    if isinstance(existing, _RepoImportGuard):
        return existing
    guard = _RepoImportGuard(repo_root)
    sys.meta_path.insert(0, guard)
    setattr(install_import_guard, _GUARD_ATTR, guard)
    return guard


def _ast_violations(repo_root: Path) -> list[str]:
    blocked = _repo_internal_modules(repo_root)
    found: list[str] = []
    for path in sorted((_package_dir(repo_root) / "chelated_kernel").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module:
                    names.append(module)
                if module.startswith("chelated_kernel"):
                    names.extend(f"{module}.{alias.name}" for alias in node.names if alias.name.startswith("_"))
            for name in names:
                if name.startswith("chelated_kernel._") or ". _" in name:
                    found.append(f"{path.name}:{name}")
                if any(part.startswith("_") for part in name.split(".")) and name.startswith("chelated_kernel."):
                    found.append(f"{path.name}:{name}")
                root = name.split(".")[0]
                if root in blocked:
                    found.append(f"{path.name}:{name}")
    return sorted(set(found))


def _stage_package(repo_root: Path, dest: Path) -> Path:
    source = _package_dir(repo_root)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "pyproject.toml", dest / "pyproject.toml")
    shutil.copyfile(source / "README.md", dest / "README.md")
    package = dest / "chelated_kernel"
    package.mkdir()
    for path in sorted((source / "chelated_kernel").glob("*.py")):
        shutil.copyfile(path, package / path.name)
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_wheel(staged: Path, dest: Path) -> tuple[Path | None, str]:
    dest.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "315532800"
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TZ"] = "UTC"
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(dest),
        str(staged),
    ]
    completed = subprocess.run(command, cwd=str(staged), env=env, capture_output=True, text=True, check=False)
    wheels = sorted(dest.glob("*.whl"))
    if completed.returncode != 0 or len(wheels) != 1:
        tail = (completed.stderr or completed.stdout or "wheel build failed")[-4000:]
        return None, tail
    return wheels[0], ""


def _wheel_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted(archive.namelist())


def _wheel_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        name = next(item for item in archive.namelist() if item.endswith(".dist-info/METADATA"))
        return archive.read(name).decode("utf-8")


def _metadata_fields(text: str) -> dict[str, list[str]]:
    header = text.split("\n\n", 1)[0]
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in header.splitlines():
        if line.startswith((" ", "\t")) and current is not None:
            fields[current][-1] = fields[current][-1] + line.strip()
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current = key.strip()
        fields.setdefault(current, []).append(value.strip())
    return fields


def _split_top(expr: str, operator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    upper = expr.upper()
    token = f" {operator} "
    index = 0
    while index < len(expr):
        char = expr[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if depth == 0 and upper.startswith(token, index):
            parts.append(expr[start:index].strip())
            index += len(token)
            start = index
            continue
        index += 1
    parts.append(expr[start:].strip())
    return [part for part in parts if part]


def _license_allowed(expression: str) -> tuple[bool, str]:
    text = " ".join(expression.replace("\n", " ").split())
    if not text:
        return False, "missing license metadata"
    lowered = text.lower()
    if "commons clause" in lowered or "commons-clause" in lowered or "sspl" in lowered:
        return False, text
    branches = _split_top(text, "OR")
    reasons: list[str] = []
    for branch in branches:
        atoms = _split_top(branch, "AND")
        branch_ok = True
        for atom in atoms:
            cleaned = atom.strip().strip("()")
            cleaned = cleaned.split(" WITH ")[0].strip()
            ident = cleaned
            upper = ident.upper()
            if upper in {"NOASSERTION", "NONE", ""}:
                branch_ok = False
                reasons.append(ident or "empty")
                continue
            if re.search(r"(^|[^A-Z])(AGPL|GPL|SSPL)([^A-Z]|$)", upper) and not upper.startswith("LGPL"):
                branch_ok = False
                reasons.append(ident)
                continue
            if ident not in ALLOWED_LICENSES:
                branch_ok = False
                reasons.append(ident)
        if branch_ok:
            return True, text
    return False, ",".join(reasons) or text


def _license_expression(fields: dict[str, list[str]], fallback_text: str = "") -> str:
    for key in ("License-Expression", "License"):
        values = [value for value in fields.get(key, []) if value and value.upper() != "UNKNOWN"]
        if values:
            return values[0]
    for classifier in fields.get("Classifier", []):
        if "MIT" in classifier:
            return "MIT"
        if "Apache" in classifier:
            return "Apache-2.0"
        if "BSD" in classifier and fallback_text:
            if "3-Clause" in fallback_text or "Revised BSD" in fallback_text:
                return "BSD-3-Clause"
            if "2-Clause" in fallback_text or "Simplified BSD" in fallback_text:
                return "BSD-2-Clause"
    if fallback_text:
        if "BSD-3-Clause" in fallback_text or "3-Clause BSD" in fallback_text:
            return "BSD-3-Clause"
        if "MIT License" in fallback_text or fallback_text.strip() == "MIT":
            return "MIT"
    return ""


def _requirement_applies(requirement: str) -> bool:
    if ";" not in requirement:
        return True
    _body, marker = requirement.split(";", 1)
    if "extra" in marker:
        return False
    try:
        from packaging.markers import Marker

        return bool(Marker(marker).evaluate())
    except Exception:
        return True


def _requirement_name(requirement: str) -> str:
    body = requirement.split(";", 1)[0]
    return re.split(r"[<>=!~\[]", body, maxsplit=1)[0].strip().replace("_", "-")


def _distribution_license(name: str) -> tuple[str, str, list[str]]:
    dist = importlib_metadata.distribution(name)
    fields: dict[str, list[str]] = {}
    for key, value in dist.metadata.items():
        fields.setdefault(key, []).append(value)
    expression = _license_expression(fields, "\n".join(fields.get("License", [])))
    requires = []
    for requirement in dist.requires or []:
        if _requirement_applies(requirement):
            cleaned = _requirement_name(requirement)
            if cleaned:
                requires.append(cleaned)
    return expression, dist.version, requires


def build_sbom(wheel_path: Path) -> dict[str, object]:
    fields = _metadata_fields(_wheel_metadata(wheel_path))
    root_name = (fields.get("Name") or ["chelated-kernel"])[0]
    root_version = (fields.get("Version") or ["0"])[0]
    root_license = _license_expression(fields)
    components: list[dict[str, object]] = []
    problems: list[str] = []
    seen: set[str] = set()

    def add_component(name: str, version: str, expression: str) -> None:
        key = name.lower().replace("_", "-")
        if key in seen:
            return
        seen.add(key)
        allowed, reason = _license_allowed(expression)
        if not allowed:
            problems.append(f"{name}: {reason}")
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "licenses": [{"license": {"id": expression or "NOASSERTION"}}],
                "purl": f"pkg:pypi/{key}@{version}",
            }
        )

    add_component(root_name, root_version, root_license)
    queue = [
        _requirement_name(requirement)
        for requirement in fields.get("Requires-Dist", [])
        if _requirement_applies(requirement)
    ]
    while queue:
        dep_name = queue.pop(0)
        key = dep_name.lower().replace("_", "-")
        if not dep_name or key in seen:
            continue
        try:
            expression, version, nested = _distribution_license(dep_name)
        except importlib_metadata.PackageNotFoundError:
            seen.add(key)
            problems.append(f"{dep_name}: missing license metadata")
            continue
        add_component(dep_name, version, expression)
        queue.extend(nested)
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": {"name": root_name, "version": root_version}},
        "components": components,
        "license_problems": problems,
    }


def _contains_magic(rows: list[list[float]]) -> bool:
    if not rows:
        return False
    array = np.asarray(rows, dtype=np.float64)
    return bool(np.any(array == MAGIC))


def _isolation_probe(repo_root: Path) -> dict[str, object]:
    _ensure_kernel_path(repo_root)
    from chelated_kernel import EventLedger, wrap_index, wrap_stream

    leaks = 0
    unauthorized = 0
    anchors, docs, queries = synthetic_pair_anchors(11, 8, 8)
    foreign_doc = list(docs[0])
    foreign_doc[0] = float(MAGIC)
    backend_a = {
        f"d{index}": {"id": f"d{index}", "vector": row, "tenant_id": "tenant-a"}
        for index, row in enumerate(docs)
    }
    backend_a[CANARY_B] = {"id": CANARY_B, "vector": foreign_doc, "tenant_id": "tenant-b"}
    anchors_b = list(anchors)
    mutated = list(anchors_b[-1][0])
    mutated[0] = float(MAGIC)
    anchors_b[-1] = (mutated, anchors_b[-1][1])
    backend_b = {"b0": {"id": "b0", "vector": foreign_doc, "tenant_id": "tenant-b"}}
    wrapped_a = wrap_index(backend_a, old_dim=8, new_dim=8, unlabeled_anchors=anchors, tenant_id="tenant-a")
    wrapped_b = wrap_index(backend_b, old_dim=8, new_dim=8, unlabeled_anchors=anchors_b, tenant_id="tenant-b")
    unauthorized += wrapped_a.unauthorized_reads + wrapped_b.unauthorized_reads
    hits = wrapped_a.search(queries[0], k=10)
    hit_ids = [hit.doc_id for hit in hits]
    if CANARY_B in hit_ids or any(CANARY_A in hit.doc_id for hit in hits):
        leaks += 1
    if _contains_magic(wrapped_a._docs.tolist()) or _contains_magic(wrapped_a._anchor_old.tolist()):
        leaks += 1
    if CANARY_B.encode("utf-8") in wrapped_a.sidecar_bytes():
        leaks += 1
    ledger_a = EventLedger(policy="sec_audit", tenant_id="tenant-a")
    ledger_b = EventLedger(policy="cve_vex", tenant_id="tenant-b")
    ledger_a.ingest(
        {
            "specversion": "1.0",
            "type": "com.chelated.sec.filing",
            "id": "a-filing",
            "source": "tenant-a",
            "time": 10,
            "data": {"statement": CANARY_A},
        }
    )
    ledger_b.ingest(
        {
            "specversion": "1.0",
            "type": "com.chelated.sec.filing",
            "id": "b-filing",
            "source": "tenant-b",
            "time": 10,
            "data": {"statement": CANARY_B},
        }
    )
    unauthorized += ledger_a.unauthorized_reads + ledger_b.unauthorized_reads
    dump_a = json.dumps(ledger_a.as_of(100, 10), sort_keys=True)
    if CANARY_B in dump_a or CANARY_B in ledger_a.non_nda_certificate():
        leaks += 1
    for receipt in ledger_a.receipts:
        if CANARY_B.encode("utf-8") in receipt.canonical_bytes():
            leaks += 1
    guard_b = wrap_stream(
        iter([CANARY_B, "tail"]),
        ledger=ledger_b,
        restricted_spans=[(CANARY_B,)],
        profile="async_armed",
        tenant_id="tenant-b",
    )
    emitted_b = list(guard_b)
    guard_a = wrap_stream(
        iter(["hello", "world"]),
        ledger=ledger_a,
        restricted_spans=[{"tokens": [CANARY_B], "tenant_id": "tenant-b"}, ("zzz",)],
        profile="strict_holdback",
        tenant_id="tenant-a",
    )
    emitted_a = list(guard_a)
    if CANARY_B in "".join(emitted_a) or CANARY_B in json.dumps(ledger_a.as_of(100, 20)):
        leaks += 1
    if CANARY_B in "".join(emitted_b):
        leaks += 1
    unauthorized += ledger_a.unauthorized_reads + ledger_b.unauthorized_reads
    certificate = ledger_a.non_nda_certificate() + "\n" + ledger_b.non_nda_certificate()
    from scripts.verify_non_nda_ip_clean import scan_text_for_ip_leaks

    ip_leaks = scan_text_for_ip_leaks(certificate, source_label="isolation-certificate")
    return {
        "unauthorized_reads": unauthorized,
        "cross_tenant_leaks": leaks,
        "ip_leaks": ip_leaks,
        "stream_a": emitted_a,
        "stream_b_suppressed": CANARY_B not in emitted_b,
    }


def _canary_probe(repo_root: Path) -> dict[str, object]:
    _ensure_kernel_path(repo_root)
    from chelated_kernel import negotiate_capability

    blocked = 0
    attempted = 0
    wrong: list[str] = []
    for name in UNSUPPORTED:
        attempted += 1
        try:
            negotiate_capability(name)
            wrong.append(name)
        except negotiate_capability.UnsupportedCapabilityError as exc:
            if exc.capability_name == name:
                blocked += 1
            else:
                wrong.append(name)
    attempted += 1
    try:
        negotiate_capability("not_a_kernel_capability")
        wrong.append("unknown")
    except negotiate_capability.UnsupportedCapabilityError:
        blocked += 1
    attempted += 1
    try:
        negotiate_capability("wrap_index", cross_tenant_raw_join=True)
        wrong.append("cross_tenant_kw")
    except negotiate_capability.UnsupportedCapabilityError:
        blocked += 1
    return {
        "attempted": attempted,
        "blocked": blocked,
        "accuracy": blocked / attempted if attempted else 0.0,
        "wrong": wrong,
        "exception_type": "UnsupportedCapabilityError",
    }


def _private_import_blocked() -> bool:
    try:
        importlib.import_module("chelated_kernel._coldstart_blocked")
    except ImportError as exc:
        return "private import blocked" in str(exc)
    return False


def _repo_import_blocked(repo_root: Path) -> bool:
    _ensure_kernel_path(repo_root)
    from chelated_kernel.profiles import probe_import_barrier

    try:
        probe_import_barrier("antigravity_engine")
    except ImportError as exc:
        return "repo-internal import blocked" in str(exc)
    return False


def _session_index(seed: int, family: str, tenant: str, foreign: str) -> dict[str, object]:
    from chelated_kernel import wrap_index

    code = FAMILIES.index(family) + 1
    old_dim, new_dim = FAMILY_DIMS[family]
    anchors, docs, queries = synthetic_pair_anchors(seed + 1009 * code, old_dim, new_dim)
    backend = {
        f"d{index}": {"id": f"d{index}", "vector": row, "tenant_id": tenant}
        for index, row in enumerate(docs)
    }
    foreign_vec = list(docs[0])
    foreign_vec[0] = float(MAGIC)
    backend[CANARY_B] = {"id": CANARY_B, "vector": foreign_vec, "tenant_id": foreign}
    wrapped = wrap_index(
        backend,
        old_dim=old_dim,
        new_dim=new_dim,
        unlabeled_anchors=anchors,
        tenant_id=tenant,
        profile="enterprise_search",
    )
    hits = wrapped.search(queries[0], k=3)
    leaks = 0
    if not hits or hits[0].doc_id != "d0":
        raise RuntimeError(f"top hit {hits[0].doc_id if hits else None}")
    if CANARY_B in [hit.doc_id for hit in hits]:
        leaks += 1
    if wrapped.unauthorized_reads != 0 or wrapped.contract.llm_write_tokens != 0:
        leaks += 1
    if wrapped.contract.confidence <= 0.99:
        raise RuntimeError(f"confidence {wrapped.contract.confidence}")
    if len(wrapped.sidecar_bytes("d0")) != 8:
        raise RuntimeError("sidecar width")
    if _contains_magic(wrapped._docs.tolist()):
        leaks += 1
    return {"cross_tenant_leaks": leaks, "confidence": wrapped.contract.confidence}


def _session_sec(seed: int, tenant: str) -> dict[str, object]:
    from chelated_kernel import EventLedger

    ledger = EventLedger(policy="sec_audit", tenant_id=tenant)
    base_time = 1000 + seed
    ledger.ingest(
        {
            "specversion": "1.0",
            "type": "com.chelated.sec.filing",
            "id": f"rev-{seed}",
            "source": tenant,
            "time": base_time,
            "data": {"statement": "original"},
        }
    )
    ledger.ingest(
        {
            "specversion": "1.0",
            "type": "com.chelated.sec.8k",
            "id": f"8k-{seed}",
            "source": tenant,
            "time": base_time + 1000,
            "data": {"item": "4.02", "supersedes": f"rev-{seed}"},
        }
    )
    early = {row["claim_id"]: row["truth"] for row in ledger.as_of(base_time + 100, 1)}
    late = {row["claim_id"]: row["truth"] for row in ledger.as_of(base_time + 1500, 2)}
    if early.get(f"rev-{seed}") != "current" or f"8k-{seed}" in early:
        raise RuntimeError(f"early snapshot {early}")
    if late.get(f"rev-{seed}") != "withdrawn" or late.get(f"8k-{seed}") != "current":
        raise RuntimeError(f"late snapshot {late}")
    if ledger.llm_write_tokens != 0 or ledger.verify_chain() != 2:
        raise RuntimeError("receipt chain")
    return {"cross_tenant_leaks": ledger.unauthorized_reads}


def _session_vex(seed: int, tenant: str) -> dict[str, object]:
    from chelated_kernel import EventLedger

    ledger = EventLedger(policy="cve_vex", tenant_id=tenant)
    vuln = f"OSV-2026-{seed}"
    ledger.ingest({"id": vuln, "affected": [{"package": {"name": "alpha"}}], "time": 10 + seed})
    ledger.ingest({"id": vuln, "affected": [{"package": {"name": "alpha"}}], "summary": "echo", "time": 11 + seed})
    clusters = ledger.echo_clusters(100, 10)
    if not any(len(cluster) == 2 for cluster in clusters):
        raise RuntimeError(f"echo clusters {clusters}")
    ledger.ingest(
        {
            "document": {"title": "vex", "category": "csaf_vex"},
            "vulnerabilities": [{"cve": vuln, "product_status": {"known_not_affected": ["prod"]}}],
            "time": 12 + seed,
        }
    )
    rows = ledger.as_of(100, 10)
    if not rows or any(row["truth"] != "disputed" for row in rows):
        raise RuntimeError(f"vex truth {[row['truth'] for row in rows]}")
    if not any(row["collapsed"] for row in rows):
        raise RuntimeError("echo member was not marked collapsed")
    return {"cross_tenant_leaks": ledger.unauthorized_reads}


def _session_stream(seed: int, tenant: str) -> dict[str, object]:
    from chelated_kernel import EventLedger, wrap_stream

    ledger = EventLedger(policy="sec_audit", tenant_id=tenant)
    profile = "async_armed" if seed % 2 else "strict_holdback"
    guard = wrap_stream(
        iter(["safe", "NO", "LEAK", "tail"]),
        ledger=ledger,
        restricted_spans=[("NO", "LEAK")],
        profile=profile,
        tenant_id=tenant,
    )
    emitted = list(guard)
    if "NO" in emitted or "LEAK" in emitted:
        raise RuntimeError(f"span leaked {emitted}")
    if "safe" not in emitted or "tail" not in emitted:
        raise RuntimeError(f"safe tokens dropped {emitted}")
    dump = json.dumps(ledger.as_of(100, 10))
    if "NO" in dump or "LEAK" in dump:
        raise RuntimeError("raw span stored in the ledger")
    if ledger.verify_chain() < 1 or ledger.llm_write_tokens != 0:
        raise RuntimeError("stream receipt")
    return {"cross_tenant_leaks": ledger.unauthorized_reads, "profile": profile}


def _run_session(repo_root: Path, family: str, vertical: str, seed: int) -> dict[str, object]:
    _ensure_kernel_path(repo_root)
    from chelated_kernel import negotiate_capability

    code = FAMILIES.index(family) + 1
    tenant = f"fam{code}s{seed}"
    foreign = f"for{code}s{seed}"
    record: dict[str, object] = {
        "family": family,
        "vertical": vertical,
        "seed": seed,
        "passed": False,
        "detail": "",
        "canary_blocked": False,
        "private_import_blocked": False,
        "cross_tenant_leaks": 0,
        "live_model_calls": 0,
    }
    try:
        if vertical == "enterprise_search_migration":
            outcome = _session_index(seed, family, tenant, foreign)
        elif vertical == "sec_item_402_restatement":
            outcome = _session_sec(seed, tenant)
        elif vertical == "osv_csaf_vex_conflict":
            outcome = _session_vex(seed, tenant)
        elif vertical == "stream_suppression_holdback":
            outcome = _session_stream(seed, tenant)
        else:
            raise RuntimeError(f"unknown vertical {vertical}")
        record["cross_tenant_leaks"] = int(outcome["cross_tenant_leaks"])
        canary = UNSUPPORTED[seed % len(UNSUPPORTED)]
        try:
            negotiate_capability(canary)
        except negotiate_capability.UnsupportedCapabilityError as exc:
            record["canary_blocked"] = exc.capability_name == canary
        record["private_import_blocked"] = _private_import_blocked()
        record["passed"] = bool(record["canary_blocked"] and record["private_import_blocked"] and record["cross_tenant_leaks"] == 0)
    except Exception as exc:
        record["detail"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
        record["passed"] = False
    return record


def wilson_upper(failures: int, total: int, z: float = 1.959963984540054) -> float:
    """One-sided Wilson upper bound on a binomial failure rate. Pure Python."""

    if total <= 0:
        return 1.0
    proportion = failures / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = proportion + z2 / (2.0 * total)
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
    return (centre + margin) / denominator


def run_gate(repo_root: Path | str, work_dir: Path | str, build_wheel: bool = True) -> dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo = Path(repo_root).resolve()
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    _ensure_kernel_path(repo)
    install_import_guard(repo)
    loc = count_loc(repo)
    readme = (_package_dir(repo) / "README.md").read_text(encoding="utf-8")
    readme_canary = CANARY_A in readme or CANARY_B in readme
    ast_hits = _ast_violations(repo)
    private_blocked = _private_import_blocked()
    repo_blocked = _repo_import_blocked(repo)
    isolation = _isolation_probe(repo)
    canaries = _canary_probe(repo)
    sessions = [_run_session(repo, family, vertical, seed) for family in FAMILIES for vertical in VERTICALS for seed in SEEDS]
    by_family: dict[str, int] = {family: 0 for family in FAMILIES}
    for session in sessions:
        if session["passed"]:
            by_family[str(session["family"])] += 1
    passed = sum(1 for session in sessions if session["passed"])
    private_successes = sum(1 for session in sessions if not session["private_import_blocked"])
    session_leaks = sum(int(session["cross_tenant_leaks"]) for session in sessions)
    leaks = int(isolation["cross_tenant_leaks"]) + session_leaks
    wheel_info: dict[str, object] = {"built": False, "deterministic": False, "sha256": "", "members_ok": False, "error": ""}
    sbom: dict[str, object] = {"components": [], "license_problems": ["wheel not built"]}
    if build_wheel:
        first = _build_wheel(_stage_package(repo, work / "stage-a"), work / "wheel-a")
        second = _build_wheel(_stage_package(repo, work / "stage-b"), work / "wheel-b")
        wheel_path, first_error = first
        _second_path, second_error = second
        if wheel_path is None or _second_path is None:
            wheel_info["error"] = first_error or second_error
        else:
            digest_a = _sha256(wheel_path)
            digest_b = _sha256(_second_path)
            members = _wheel_members(wheel_path)
            members_ok = all("scripts/" not in name and "test_" not in Path(name).name for name in members)
            members_ok = members_ok and any(name.endswith("chelated_kernel/__init__.py") for name in members)
            wheel_info = {
                "built": True,
                "deterministic": digest_a == digest_b,
                "sha256": digest_a,
                "sha256_repeat": digest_b,
                "members_ok": members_ok,
                "member_count": len(members),
                "error": "",
            }
            sbom = build_sbom(wheel_path)
            (work / "sbom.cdx.json").write_text(json.dumps(sbom, indent=2, sort_keys=True), encoding="utf-8")
    family_ok = all(count >= 8 for count in by_family.values())
    license_ok = not sbom.get("license_problems")
    gate_pass = bool(
        loc < LOC_LIMIT
        and not readme_canary
        and not ast_hits
        and private_blocked
        and repo_blocked
        and int(isolation["unauthorized_reads"]) == 0
        and leaks == 0
        and not isolation["ip_leaks"]
        and float(canaries["accuracy"]) == 1.0
        and passed >= 30
        and family_ok
        and private_successes == 0
        and (not build_wheel or (wheel_info["built"] and wheel_info["deterministic"] and wheel_info["members_ok"] and license_ok))
    )
    report: dict[str, object] = {
        "bundle": 49,
        "execution": "deterministic-local-smoke",
        "live_model_calls": 0,
        "loc": loc,
        "loc_limit": LOC_LIMIT,
        "readme_canary": readme_canary,
        "wheel": wheel_info,
        "sbom_components": len(sbom.get("components", [])),
        "license_problems": sbom.get("license_problems", []),
        "unauthorized_reads": isolation["unauthorized_reads"],
        "cross_tenant_leaks": leaks,
        "ip_leaks": isolation["ip_leaks"],
        "canary": canaries,
        "private_import_blocked": private_blocked,
        "repo_import_blocked": repo_blocked,
        "ast_violations": ast_hits,
        "sessions": len(sessions),
        "sessions_passed": passed,
        "per_family_passed": by_family,
        "private_import_successes": private_successes,
        "failure_rate_wilson_upper_95": wilson_upper(len(sessions) - passed, len(sessions)),
        "pass_rule": {
            "min_overall": 30,
            "overall_sessions": 36,
            "min_per_family": 8,
            "family_sessions": 12,
            "private_imports": 0,
            "canary_accuracy": 1.0,
            "cross_tenant_leaks": 0,
        },
        "gate_pass": gate_pass,
        "session_records": sessions,
    }
    (work / "gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Bundle 49 chelated-kernel cold-start gate.")
    parser.add_argument("--out", required=True, help="Directory for the wheel, SBOM, and gate report.")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    report = run_gate(repo, Path(args.out), build_wheel=True)
    print(f"gate_pass={report['gate_pass']} loc={report['loc']} sessions={report['sessions_passed']}/36")
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
