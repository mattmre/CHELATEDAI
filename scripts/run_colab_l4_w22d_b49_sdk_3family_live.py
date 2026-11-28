#!/usr/bin/env python3
"""Colab L4 Wave 22D: Bundle #49 Live 3-Family `chelated-kernel` Cold-Start Benchmark (36 Live Sessions).

Builds the reproducible `chelated-kernel` wheel cleanly inside the Colab L4 VM, installs it into an
isolated `--target` directory with ZERO repository access on `sys.path`, and runs 3 genuinely unrelated
open-weight model families (`Qwen/Qwen2.5-3B-Instruct`, `microsoft/Phi-3.5-mini-instruct`,
`ibm-granite/granite-3.1-2b-instruct`) x 4 verticals x 3 seeds = 36 live-model cold-start sessions:
  - Each model sees ONLY the sanitized public `packages/chelated_kernel/README.md` (with internal
    verification script mentions stripped) plus the task specification for one vertical.
  - Generated Python adapter code is extracted, AST-audited for zero private/repo imports, and
    executed in an isolated child subprocess (`PYTHONPATH=<wheel_target>`, `cwd=/tmp/...`) against
    real functional assertions, 2-tenant canary isolation, and unsupported-capability canaries.
  - Enforces Bundle #49 pass rules: >= 30/36 overall, >= 8/12 per family, 0 private imports,
    100% unsupported-capability canary blocking, and 0 cross-tenant leaks.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

LIVE_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct"),
    ("granite-3.1-2b", "ibm-granite/granite-3.1-2b-instruct"),
)
VERTICALS: Tuple[str, ...] = (
    "enterprise_search_migration",
    "sec_item_402_restatement",
    "osv_csaf_vex_conflict",
    "stream_suppression_holdback",
)
SEEDS: Tuple[int, ...] = (0, 1, 2)
UNSUPPORTED_CANARIES: Tuple[str, ...] = (
    "bare_metal_p4_tcam_asic_dataplane",
    "physical_weight_erasure_unlearning",
    "hdd_servo_mcu_firmware",
    "cross_tenant_raw_join",
)
CANARY_B = "PLANTED-CANARY-TENANT-B-4d2a"
_EXTRA_BLOCKED_ROOTS = frozenset(
    {"egv", "demo_workbench", "computational_storage_poc", "formal", "scripts", "tests", "rlm_reference", "chelated_pv7_spine"}
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_upper(failures: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 1.0
    p = failures / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
    return (centre + margin) / denom


def sanitized_public_readme(repo_root: Path) -> str:
    raw = (repo_root / "packages" / "chelated_kernel" / "README.md").read_text(encoding="utf-8")
    # Strip final Build/verifier paragraph that mentions internal scripts/ paths so models see pure public docs
    lines = [ln for ln in raw.splitlines() if "scripts/" not in ln and "run_pv7_w17_6d" not in ln]
    return "\n".join(lines).strip()


def build_and_install_isolated_wheel(repo_root: Path, work_dir: Path) -> Tuple[Path, Path, str, bool]:
    """Build chelated-kernel wheel twice (determinism check) and install into an isolated --target dir."""
    pkg_src = repo_root / "packages" / "chelated_kernel"
    digests: List[str] = []
    wheel_path: Optional[Path] = None
    for tag in ("a", "b"):
        staged = work_dir / f"stage-{tag}"
        dest = work_dir / f"wheel-{tag}"
        if staged.exists():
            shutil.rmtree(staged)
        if dest.exists():
            shutil.rmtree(dest)
        staged.mkdir(parents=True)
        dest.mkdir(parents=True)
        shutil.copyfile(pkg_src / "pyproject.toml", staged / "pyproject.toml")
        shutil.copyfile(pkg_src / "README.md", staged / "README.md")
        shutil.copytree(
            pkg_src / "chelated_kernel",
            staged / "chelated_kernel",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        env = dict(os.environ)
        env.update({"SOURCE_DATE_EPOCH": "315532800", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "TZ": "UTC"})
        cmd = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(dest), str(staged)]
        proc = subprocess.run(cmd, cwd=str(staged), env=env, capture_output=True, text=True, check=False)
        wheels = sorted(dest.glob("*.whl"))
        if proc.returncode != 0 or len(wheels) != 1:
            raise RuntimeError(f"Wheel build failed ({tag}): {(proc.stderr or proc.stdout)[-2000:]}")
        wheel_path = wheels[0]
        digests.append(sha256_file(wheel_path))

    assert wheel_path is not None
    target_dir = work_dir / "isolated_wheel_target"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    inst = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(target_dir), str(wheel_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if inst.returncode != 0:
        raise RuntimeError(f"Wheel install failed: {(inst.stderr or inst.stdout)[-2000:]}")
    return wheel_path, target_dir, digests[0], (digests[0] == digests[1])


def vertical_task_prompt(vertical: str, seed: int, canary_cap: str) -> str:
    common_rules = (
        "Write ONLY a single Python code block defining `def run_adapter(ctx: dict) -> dict:`.\n"
        "Strict rules:\n"
        "- Import ONLY from `chelated_kernel` public exports (`wrap_index`, `EventLedger`, `wrap_stream`, `negotiate_capability`) and Python stdlib.\n"
        "- NEVER import private submodules (`chelated_kernel._*`) or external packages.\n"
        f"- Inside `run_adapter(ctx)`, call `negotiate_capability({canary_cap!r})` inside a `try/except negotiate_capability.UnsupportedCapabilityError` block and set `result['canary_blocked'] = True` ONLY if that exact exception is caught.\n"
        "- Do not include markdown prose outside the ```python ... ``` code block.\n"
    )
    if vertical == "enterprise_search_migration":
        return (
            common_rules
            + "Task (`enterprise_search_migration`):\n"
            "Given `ctx` containing `backend` (dict of doc records with `tenant_id`), `old_dim` (int), `new_dim` (int), "
            "`anchors` (list of `(old_vec, new_vec)` pairs), `query` (list of float), and `tenant_id` (str):\n"
            "1. Wrap `backend` via `wrap_index(ctx['backend'], old_dim=ctx['old_dim'], new_dim=ctx['new_dim'], unlabeled_anchors=ctx['anchors'], tenant_id=ctx['tenant_id'], profile='enterprise_search')`.\n"
            "2. Run `hits = wrapped.search(ctx['query'], k=3)`.\n"
            "3. Return `{'top_doc_id': hits[0].doc_id, 'hit_ids': [h.doc_id for h in hits], 'unauthorized_reads': wrapped.unauthorized_reads, 'sidecar_len': len(wrapped.sidecar_bytes(hits[0].doc_id)), 'canary_blocked': canary_blocked}`."
        )
    if vertical == "sec_item_402_restatement":
        return (
            common_rules
            + "Task (`sec_item_402_restatement`):\n"
            "Given `ctx` containing `tenant_id`, `filing_event` (CloudEvent `com.chelated.sec.filing`), "
            "`item402_event` (CloudEvent `com.chelated.sec.8k` with item `4.02` superseding the filing), `early_valid_time`, and `late_valid_time`:\n"
            "1. Create `ledger = EventLedger(policy='sec_audit', tenant_id=ctx['tenant_id'])`.\n"
            "2. Ingest `ctx['filing_event']`, then ingest `ctx['item402_event']`.\n"
            "3. Query `early = {r['claim_id']: r['truth'] for r in ledger.as_of(ctx['early_valid_time'], 1)}` and `late = {r['claim_id']: r['truth'] for r in ledger.as_of(ctx['late_valid_time'], 2)}`.\n"
            "4. Call `chain_len = ledger.verify_chain()`.\n"
            "5. Return `{'early': early, 'late': late, 'chain_len': chain_len, 'unauthorized_reads': ledger.unauthorized_reads, 'canary_blocked': canary_blocked}`."
        )
    if vertical == "osv_csaf_vex_conflict":
        return (
            common_rules
            + "Task (`osv_csaf_vex_conflict`):\n"
            "Given `ctx` containing `tenant_id`, `osv_origin`, `osv_echo`, and `csaf_vex`:\n"
            "1. Create `ledger = EventLedger(policy='cve_vex', tenant_id=ctx['tenant_id'])`.\n"
            "2. Ingest `ctx['osv_origin']`, `ctx['osv_echo']`, and `ctx['csaf_vex']` in order.\n"
            "3. Call `rows = ledger.as_of(100, 10)`.\n"
            "4. Return `{'truths': [r['truth'] for r in rows], 'any_collapsed': any(r.get('collapsed') for r in rows), 'unauthorized_reads': ledger.unauthorized_reads, 'canary_blocked': canary_blocked}`."
        )
    if vertical == "stream_suppression_holdback":
        return (
            common_rules
            + "Task (`stream_suppression_holdback`):\n"
            "Given `ctx` containing `tenant_id`, `tokens` (list of str), `restricted_spans` (list of tuple of str), and `profile` (`strict_holdback` or `async_armed`):\n"
            "1. Create `ledger = EventLedger(policy='sec_audit', tenant_id=ctx['tenant_id'])`.\n"
            "2. Create `guard = wrap_stream(iter(ctx['tokens']), ledger=ledger, restricted_spans=ctx['restricted_spans'], profile=ctx['profile'], tenant_id=ctx['tenant_id'])`.\n"
            "3. Collect `emitted = list(guard)` and `chain_len = ledger.verify_chain()`.\n"
            "4. Return `{'emitted': emitted, 'suppressed_spans': guard.suppressed_spans, 'chain_len': chain_len, 'unauthorized_reads': ledger.unauthorized_reads, 'canary_blocked': canary_blocked}`."
        )
    raise ValueError(f"Unknown vertical={vertical!r}")


def extract_python_code(raw_text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: if model emitted raw python without fences starting at import/def
    idx_def = raw_text.find("def run_adapter")
    idx_imp = raw_text.find("from chelated_kernel")
    start = min([i for i in (idx_def, idx_imp) if i >= 0], default=0)
    return raw_text[start:].strip()


def audit_generated_ast(code: str, blocked_repo_stems: frozenset[str]) -> List[str]:
    violations: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]
    for node in ast.walk(tree):
        names: List[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod:
                names.append(mod)
            if mod.startswith("chelated_kernel"):
                names.extend(f"{mod}.{alias.name}" for alias in node.names if alias.name.startswith("_"))
        for name in names:
            if name.startswith("chelated_kernel._") or (
                name.startswith("chelated_kernel.") and any(p.startswith("_") for p in name.split("."))
            ):
                violations.append(f"private_import:{name}")
            if name.split(".")[0] in blocked_repo_stems:
                violations.append(f"repo_import:{name}")
    return sorted(set(violations))


def verify_adapter_in_isolated_subprocess(
    *,
    code: str,
    vertical: str,
    seed: int,
    wheel_target_dir: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    """Execute generated `run_adapter` inside an isolated subprocess with ONLY the built wheel on PYTHONPATH."""
    harness = r'''
import json, sys
from pathlib import Path
import numpy as np

wheel_target = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
vertical = sys.argv[3]
seed = int(sys.argv[4])
code_path = Path(sys.argv[5])

import chelated_kernel
origin = Path(chelated_kernel.__file__).resolve()
assert origin.is_relative_to(wheel_target), f"imported outside wheel target: {origin}"
assert not any(Path(p).resolve() == repo_root for p in sys.path if p), "repo root leaked onto sys.path"

ns = {}
exec(compile(code_path.read_text(encoding="utf-8"), str(code_path), "exec"), ns, ns)
fn = ns.get("run_adapter")
if not callable(fn):
    raise RuntimeError("run_adapter(ctx) not defined")

CANARY_B = "PLANTED-CANARY-TENANT-B-4d2a"
tenant = f"live-t-{seed}"
foreign = f"live-for-{seed}"

if vertical == "enterprise_search_migration":
    rng = np.random.default_rng(1009 + seed)
    old_dim, new_dim = 8, 8
    q_mat = np.linalg.qr(rng.normal(size=(new_dim, old_dim)))[0]
    old = rng.normal(size=(64, old_dim))
    fit = old[:48] - old[:48].mean(axis=0, keepdims=True)
    chol = np.linalg.cholesky(fit.T @ fit)
    old[:48] = fit @ np.linalg.inv(chol).T
    new = old @ q_mat.T
    docs = np.eye(old_dim, dtype=np.float64)[:5] * 5.0
    query = (docs[0] @ q_mat.T).tolist()
    backend = {f"d{i}": {"id": f"d{i}", "vector": docs[i].tolist(), "tenant_id": tenant} for i in range(5)}
    fvec = docs[0].tolist()
    fvec[0] = 777777.25
    backend[CANARY_B] = {"id": CANARY_B, "vector": fvec, "tenant_id": foreign}
    anchors = [(old[i].tolist(), new[i].tolist()) for i in range(64)]
    ctx = {"backend": backend, "old_dim": old_dim, "new_dim": new_dim, "anchors": anchors, "query": query, "tenant_id": tenant}
    out = fn(ctx)
    leaks = int(CANARY_B in (out.get("hit_ids") or [])) + int(out.get("unauthorized_reads", 0) != 0)
    ok = (out.get("top_doc_id") == "d0" and int(out.get("sidecar_len", 0)) == 8 and leaks == 0)
elif vertical == "sec_item_402_restatement":
    base_t = 1000 + seed
    ctx = {
        "tenant_id": tenant,
        "filing_event": {"specversion": "1.0", "type": "com.chelated.sec.filing", "id": f"rev-{seed}", "source": tenant, "time": base_t, "data": {"statement": "orig"}},
        "item402_event": {"specversion": "1.0", "type": "com.chelated.sec.8k", "id": f"8k-{seed}", "source": tenant, "time": base_t + 1000, "data": {"item": "4.02", "supersedes": f"rev-{seed}"}},
        "early_valid_time": base_t + 100,
        "late_valid_time": base_t + 1500,
    }
    out = fn(ctx)
    leaks = int(out.get("unauthorized_reads", 0) != 0)
    early = out.get("early") or {}
    late = out.get("late") or {}
    ok = (
        early.get(f"rev-{seed}") == "current"
        and f"8k-{seed}" not in early
        and late.get(f"rev-{seed}") == "withdrawn"
        and late.get(f"8k-{seed}") == "current"
        and int(out.get("chain_len", 0)) == 2
        and leaks == 0
    )
elif vertical == "osv_csaf_vex_conflict":
    vuln = f"OSV-2026-{seed}"
    ctx = {
        "tenant_id": tenant,
        "osv_origin": {"id": vuln, "affected": [{"package": {"name": "alpha"}}], "time": 10 + seed},
        "osv_echo": {"id": vuln, "affected": [{"package": {"name": "alpha"}}], "summary": "echo", "time": 11 + seed},
        "csaf_vex": {"document": {"title": "vex", "category": "csaf_vex"}, "vulnerabilities": [{"cve": vuln, "product_status": {"known_not_affected": ["prod"]}}], "time": 12 + seed},
    }
    out = fn(ctx)
    leaks = int(out.get("unauthorized_reads", 0) != 0)
    truths = out.get("truths") or []
    ok = bool(truths) and all(t == "disputed" for t in truths) and bool(out.get("any_collapsed")) and leaks == 0
elif vertical == "stream_suppression_holdback":
    ctx = {
        "tenant_id": tenant,
        "tokens": ["safe", "NO", "NO", "LEAK", "tail"],
        "restricted_spans": [("NO", "LEAK")],
        "profile": "async_armed" if seed % 2 else "strict_holdback",
    }
    out = fn(ctx)
    leaks = int(out.get("unauthorized_reads", 0) != 0)
    emitted = out.get("emitted") or []
    ok = (emitted == ["safe", "NO", "tail"] and int(out.get("suppressed_spans", 0)) == 1 and int(out.get("chain_len", 0)) >= 1 and leaks == 0)
else:
    raise RuntimeError(f"Unknown vertical {vertical}")

print(json.dumps({
    "functional_ok": bool(ok),
    "canary_blocked": bool(out.get("canary_blocked", False)),
    "cross_tenant_leaks": int(leaks),
    "origin": str(origin),
}))
'''
    with tempfile.TemporaryDirectory(prefix="b49-live-exec-") as td:
        tpath = Path(td)
        code_file = tpath / "generated_adapter.py"
        code_file.write_text(code, encoding="utf-8")
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "CUDA_VISIBLE_DEVICES"}
        }
        clean_env["PYTHONPATH"] = str(wheel_target_dir)
        clean_env["CUDA_VISIBLE_DEVICES"] = ""
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", harness, str(wheel_target_dir), str(repo_root.resolve()), vertical, str(seed), str(code_file)],
            cwd=str(tpath),
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "functional_ok": False,
                "canary_blocked": False,
                "cross_tenant_leaks": 0,
                "error": (proc.stderr or proc.stdout or "nonzero exit")[-800:],
            }
        last_line = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1]
        parsed = json.loads(last_line)
        parsed["error"] = ""
        return parsed


def run_benchmark_with_generator(
    *,
    repo_root: Path,
    work_dir: Path,
    families: Sequence[Tuple[str, str]],
    generate_fn: Callable[[str, str, str, int], str],
) -> Dict[str, Any]:
    wheel_path, target_dir, wheel_sha, wheel_det = build_and_install_isolated_wheel(repo_root, work_dir)
    readme_text = sanitized_public_readme(repo_root)
    blocked_stems = _EXTRA_BLOCKED_ROOTS | {p.stem for p in repo_root.glob("*.py")}

    sessions: List[Dict[str, Any]] = []
    per_family_passed: Dict[str, int] = {fam: 0 for fam, _ in families}

    for fam_name, model_id in families:
        for vertical in VERTICALS:
            for seed in SEEDS:
                canary_cap = UNSUPPORTED_CANARIES[seed % len(UNSUPPORTED_CANARIES)]
                prompt = vertical_task_prompt(vertical, seed, canary_cap)
                t_s = time.perf_counter()
                raw_reply = generate_fn(fam_name, model_id, f"# Public SDK Documentation\n\n{readme_text}\n\n# Task\n\n{prompt}", seed)
                gen_s = round(time.perf_counter() - t_s, 2)
                code = extract_python_code(raw_reply)
                ast_viols = audit_generated_ast(code, blocked_stems)
                if ast_viols:
                    exec_res = {
                        "functional_ok": False,
                        "canary_blocked": False,
                        "cross_tenant_leaks": 0,
                        "error": f"AST violations: {ast_viols}",
                    }
                else:
                    exec_res = verify_adapter_in_isolated_subprocess(
                        code=code,
                        vertical=vertical,
                        seed=seed,
                        wheel_target_dir=target_dir,
                        repo_root=repo_root,
                    )
                passed = bool(
                    not ast_viols
                    and exec_res["functional_ok"]
                    and exec_res["canary_blocked"]
                    and int(exec_res["cross_tenant_leaks"]) == 0
                )
                if passed:
                    per_family_passed[fam_name] += 1
                sessions.append(
                    {
                        "family": fam_name,
                        "model_id": model_id,
                        "vertical": vertical,
                        "seed": seed,
                        "canary_capability": canary_cap,
                        "live_model_calls": 1,
                        "generation_seconds": gen_s,
                        "code_sha256": sha256_bytes(code.encode("utf-8")),
                        "ast_violations": ast_viols,
                        "functional_ok": bool(exec_res["functional_ok"]),
                        "canary_blocked": bool(exec_res["canary_blocked"]),
                        "cross_tenant_leaks": int(exec_res["cross_tenant_leaks"]),
                        "passed": passed,
                        "error": exec_res.get("error", ""),
                    }
                )
                print(
                    f"[w22d] {fam_name} | {vertical} | s={seed} -> passed={passed} "
                    f"(func={exec_res['functional_ok']} canary={exec_res['canary_blocked']} gen={gen_s}s)"
                    + (f" err={exec_res['error'][:120]}" if not passed else ""),
                    flush=True,
                )

    total = len(sessions)
    passed_total = sum(1 for s in sessions if s["passed"])
    private_imports_total = sum(len(s["ast_violations"]) for s in sessions)
    leaks_total = sum(int(s["cross_tenant_leaks"]) for s in sessions)
    canary_acc = sum(1 for s in sessions if s["canary_blocked"]) / float(max(total, 1))
    family_ok = all(cnt >= 8 for cnt in per_family_passed.values())
    gate_pass = bool(
        wheel_det
        and passed_total >= 30
        and family_ok
        and private_imports_total == 0
        and leaks_total == 0
        and canary_acc == 1.0
    )
    return {
        "bundle_id": 49,
        "item": "6D",
        "wheel_sha256": wheel_sha,
        "wheel_deterministic": wheel_det,
        "live_model_calls": total,
        "sessions_total": total,
        "sessions_passed": passed_total,
        "per_family_passed": per_family_passed,
        "private_import_violations_total": private_imports_total,
        "cross_tenant_leaks_total": leaks_total,
        "canary_accuracy": round(canary_acc, 6),
        "failure_rate_wilson_upper_95": round(wilson_upper(total - passed_total, total), 6),
        "pass_rule": {"min_overall": 30, "min_per_family": 8, "private_imports": 0, "canary_accuracy": 1.0, "leaks": 0},
        "gate_pass": gate_pass,
        "sessions": sessions,
    }


def run_self_test() -> Dict[str, Any]:
    """Deterministic CPU self-test verifying wheel build, isolated subprocess execution, AST guard, and scoring."""
    def _mock_gen(fam: str, mid: str, prompt: str, seed: int) -> str:
        canary = UNSUPPORTED_CANARIES[seed % len(UNSUPPORTED_CANARIES)]
        if "enterprise_search_migration" in prompt:
            return f"""```python
from chelated_kernel import wrap_index, negotiate_capability
def run_adapter(ctx: dict) -> dict:
    canary_blocked = False
    try:
        negotiate_capability({canary!r})
    except negotiate_capability.UnsupportedCapabilityError:
        canary_blocked = True
    wrapped = wrap_index(ctx['backend'], old_dim=ctx['old_dim'], new_dim=ctx['new_dim'], unlabeled_anchors=ctx['anchors'], tenant_id=ctx['tenant_id'], profile='enterprise_search')
    hits = wrapped.search(ctx['query'], k=3)
    return {{'top_doc_id': hits[0].doc_id, 'hit_ids': [h.doc_id for h in hits], 'unauthorized_reads': wrapped.unauthorized_reads, 'sidecar_len': len(wrapped.sidecar_bytes(hits[0].doc_id)), 'canary_blocked': canary_blocked}}
```"""
        if "sec_item_402_restatement" in prompt:
            return f"""```python
from chelated_kernel import EventLedger, negotiate_capability
def run_adapter(ctx: dict) -> dict:
    canary_blocked = False
    try:
        negotiate_capability({canary!r})
    except negotiate_capability.UnsupportedCapabilityError:
        canary_blocked = True
    ledger = EventLedger(policy='sec_audit', tenant_id=ctx['tenant_id'])
    ledger.ingest(ctx['filing_event'])
    ledger.ingest(ctx['item402_event'])
    early = {{r['claim_id']: r['truth'] for r in ledger.as_of(ctx['early_valid_time'], 1)}}
    late = {{r['claim_id']: r['truth'] for r in ledger.as_of(ctx['late_valid_time'], 2)}}
    return {{'early': early, 'late': late, 'chain_len': ledger.verify_chain(), 'unauthorized_reads': ledger.unauthorized_reads, 'canary_blocked': canary_blocked}}
```"""
        if "osv_csaf_vex_conflict" in prompt:
            return f"""```python
from chelated_kernel import EventLedger, negotiate_capability
def run_adapter(ctx: dict) -> dict:
    canary_blocked = False
    try:
        negotiate_capability({canary!r})
    except negotiate_capability.UnsupportedCapabilityError:
        canary_blocked = True
    ledger = EventLedger(policy='cve_vex', tenant_id=ctx['tenant_id'])
    ledger.ingest(ctx['osv_origin'])
    ledger.ingest(ctx['osv_echo'])
    ledger.ingest(ctx['csaf_vex'])
    rows = ledger.as_of(100, 10)
    return {{'truths': [r['truth'] for r in rows], 'any_collapsed': any(r.get('collapsed') for r in rows), 'unauthorized_reads': ledger.unauthorized_reads, 'canary_blocked': canary_blocked}}
```"""
        return f"""```python
from chelated_kernel import EventLedger, wrap_stream, negotiate_capability
def run_adapter(ctx: dict) -> dict:
    canary_blocked = False
    try:
        negotiate_capability({canary!r})
    except negotiate_capability.UnsupportedCapabilityError:
        canary_blocked = True
    ledger = EventLedger(policy='sec_audit', tenant_id=ctx['tenant_id'])
    guard = wrap_stream(iter(ctx['tokens']), ledger=ledger, restricted_spans=ctx['restricted_spans'], profile=ctx['profile'], tenant_id=ctx['tenant_id'])
    emitted = list(guard)
    return {{'emitted': emitted, 'suppressed_spans': guard.suppressed_spans, 'chain_len': ledger.verify_chain(), 'unauthorized_reads': ledger.unauthorized_reads, 'canary_blocked': canary_blocked}}
```"""

    with tempfile.TemporaryDirectory(prefix="w22d-selftest-") as td:
        res = run_benchmark_with_generator(
            repo_root=REPO_ROOT,
            work_dir=Path(td),
            families=(("mock-fam-1", "mock/1"),),
            generate_fn=_mock_gen,
        )
    assert res["sessions_passed"] == 12 and res["wheel_deterministic"]
    return {"self_test": "PASS", "sessions_passed": res["sessions_passed"], "wheel_sha256": res["wheel_sha256"][:12]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("/content/out/pv7_w22d_b49_sdk_3family_live"))
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--source-archive-sha256", type=str, default=os.environ.get("CHELATED_SRC_TAR_SHA256", "unknown"))
    args = ap.parse_args(argv)

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Wave 22D live 3-family cold-start benchmark")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    vram_mib = round(float(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2)), 2)
    hardware_cohort = f"colab:{gpu_name}"
    print(f"[w22d] hardware_cohort={hardware_cohort} vram_mib={vram_mib} torch={torch.__version__}", flush=True)

    loaded_cache: Dict[str, Any] = {"model_id": None, "model": None, "tok": None}

    def _live_generate(fam: str, model_id: str, full_prompt: str, seed: int) -> str:
        if loaded_cache["model_id"] != model_id:
            if loaded_cache["model"] is not None:
                del loaded_cache["model"], loaded_cache["tok"]
                loaded_cache["model"] = None
                loaded_cache["tok"] = None
                gc.collect()
                torch.cuda.empty_cache()
            print(f"[w22d] Loading live family {fam} ({model_id}) in bfloat16 ...", flush=True)
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
            mdl.eval()
            loaded_cache.update({"model_id": model_id, "model": mdl, "tok": tok})

        tok = loaded_cache["tok"]
        mdl = loaded_cache["model"]
        messages = [
            {
                "role": "system",
                "content": "You are a precise Python SDK engineer. Output ONLY one ```python ... ``` block implementing run_adapter(ctx) strictly from the provided documentation.",
            },
            {"role": "user", "content": full_prompt},
        ]
        chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(chat_text, return_tensors="pt").to("cuda")
        torch.manual_seed(4900 + seed)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": 512,
            "pad_token_id": tok.eos_token_id,
        }
        if seed == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.15
            gen_kwargs["top_p"] = 0.9
        with torch.no_grad():
            out_ids = mdl.generate(**inputs, **gen_kwargs)
        new_ids = out_ids[0][inputs["input_ids"].shape[1] :]
        return str(tok.decode(new_ids, skip_special_tokens=True))

    with tempfile.TemporaryDirectory(prefix="w22d-wheel-") as td:
        bench = run_benchmark_with_generator(
            repo_root=REPO_ROOT,
            work_dir=Path(td),
            families=LIVE_FAMILIES,
            generate_fn=_live_generate,
        )

    if loaded_cache["model"] is not None:
        del loaded_cache["model"], loaded_cache["tok"]
        gc.collect()
        torch.cuda.empty_cache()

    wall_total = round(time.perf_counter() - t0, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "colab_l4_w22d_b49_sdk_3family_live_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit_sha": args.git_commit_sha,
            "source_archive_sha256": args.source_archive_sha256,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "hardware_cohort": hardware_cohort,
            "is_local_dgx_spark_evidence": False,
            "blind_fixtures_touched": False,
            "gpu_name": gpu_name,
            "gpu_total_vram_mib": vram_mib,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "wall_seconds_total": wall_total,
        },
        "benchmark": bench,
    }
    m_path = args.output_dir / "w22d_b49_sdk_3family_manifest.json"
    m_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    m_path.write_bytes(m_bytes)
    m_sha = sha256_bytes(m_bytes)

    receipt = {
        "schema_version": "colab_l4_burst_receipt_v1",
        "shard_id": "w22d_b49_sdk_3family_live",
        "hardware_cohort": hardware_cohort,
        "git_commit_sha": args.git_commit_sha,
        "source_archive_sha256": args.source_archive_sha256,
        "manifest_file": m_path.name,
        "manifest_sha256": m_sha,
        "exit_code": 0,
        "wall_seconds_total": wall_total,
    }
    (args.output_dir / "w22d_b49_sdk_3family_receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        f"[w22d] DONE sha256={m_sha[:12]} passed={bench['sessions_passed']}/36 "
        f"per_family={bench['per_family_passed']} gate_pass={bench['gate_pass']} wall={wall_total}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
