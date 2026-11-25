"""Single-command offline verifier (`scripts/verify_all_offline.py`).

Implements wave14-r3-final-consensus-master-plan.md sections 1.3 / 3.1 / 5.

Modes
-----
``--fast`` (default)
    Zero network / GPU / Docker.  Target: well under 45 s of wall-clock on a laptop CPU.
    Checks, in order:

    a. Runtime / platform / Python provenance plus the explicit formal-tier disclosure
       ``"formal_tier": "SKIPPED_USE_FULL_FORMAL"`` and the list of skipped checks.
    b. Every bundle in ``docs/research/bundle_registry_spec.json`` (any count: 40, 45, 46, ...):
       manifest + receipt exist, receipt ``manifest_sha256`` equals the on-disk manifest bytes,
       ancillary JSON and ``artifact_hashes`` / ``artifact_sha256`` sidecars exist and match;
       gitignored raw ``.csv`` / ``.jsonl`` are optional offline but hashed when present;
       ``*sealed_key*`` files are never opened (existence only).  The standalone Part VII
       bundles D6 / D8 are always required; ``wave11_d7_`` / ``wave11_d9_`` / ``wave11_d10_``
       and ``wave14_d11_`` (the Wave 14b PM6 takedown carry-forward) are verified whenever
       their ``report_receipt.json`` exists.
    c. D8 Compliance Copilot receipts: Ed25519 signature over every ``entry_hash`` plus the
       SHA-256 hash chain from the manifest, then a live deterministic replay of the engine
       (in a subprocess) that must reproduce the sealed entry hashes byte for byte.
    c2. D11 (when ``artifacts/wave14_d11_*`` exists): all 12 ``copilot_receipt_v2`` entries
       re-verified from first principles (``entry_hash`` recomputed from
       ``prev_hash || canonical_json(payload)``, Ed25519 signature, ``seq`` / ``prev_hash``
       linkage, integer-only payloads, head == ``report_receipt.receipts_head_hash`` ==
       manifest ref, ``non_nda_leak_hits == 0`` plus a live scanner rescan).  ``metrics.d11``
       recomputes the headline-arm ``r5_benign_fpr_le_002`` TEST target and the 12-test Holm
       step-down straight from the sealed manifest and fails unless
       ``pm6_posthoc_statistical_reanalysis.json`` binds the manifest sha256, discloses the
       target as FAILED, matches Holm, and carries disclosures D1-D11; the addendum must also
       be byte-identical to a fresh CPU recomputation.
    c3. Formal spec lock (when ``formal/spec_sha256.json`` exists): ``Spec.lean`` sha256 equals
       the pinned lock and ``conformance_report.json`` keeps
       ``incremental_with_deletions_status == "conformance_tested_unproven_in_lean"``.
    d. The 56-pattern Non-NDA scanner over the teaser markdown, the teaser deck HTML and the
       Compliance Copilot page: 0 hits each.
    e. Headline-metric cross-checks straight from the committed manifests (D6 cross-engine
       nDCG@10 parity and 0.00 % stale resurrection at 0/10/50/90 %, D8 38/38 checks and
       0/56 Non-NDA hits) and ``verified_registry.json`` self-integrity.
    f. ``release_status`` honesty gate: PRE-RELEASE is retained while PM-4b human grading is
       pending.

``--full-formal``
    Runs everything above, then the formal gates (``lake build``, 0 ``sorry`` /
    ``native_decide``, ``#print axioms`` allowlist with negative control, optional external
    kernel, EK5 conformance).  If ``formal/`` (``lakefile.lean`` / ``lakefile.toml``),
    ``lake``, the Mathlib ``.olean`` cache, ``nanoda_bin``, ``formal/run_conformance.py`` or
    (when the export is missing/stale) ``lean4export`` is missing, the formal tier is reported
    as ``NOT_RUN`` and the exit code is 2.  A missing toolchain is never reported as PASS or
    PASS_PARTIAL.  The external kernel must exit 0 AND print ``Checked N declarations with no
    typechecker errors`` with ``N > 0`` and ``N >= len(export_declarations)``.

Exit codes: 0 = everything requested passed, 1 = at least one check failed, 2 = the
requested formal tier could not run (and nothing else failed).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib.util
import json
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent

SPEC_REL = "docs/research/bundle_registry_spec.json"
REGISTRY_REL = "docs/research/verified_registry.json"

FORMAL_TIER_SKIPPED = "SKIPPED_USE_FULL_FORMAL"
FORMAL_TIER_NOT_RUN = "NOT_RUN"
FORMAL_TIER_PASS = "PASS"
FORMAL_TIER_PASS_PARTIAL = "PASS_PARTIAL"
FORMAL_TIER_FAIL = "FAIL"

FAST_SKIPPED_CHECKS: Tuple[str, ...] = (
    "lean_lake_build",
    "print_axioms_allowlist",
    "nanoda_external_kernel",
    "ek5_1000_mutation_conformance",
)

ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
GENESIS_HASH = "0" * 64
EXPECTED_PATTERN_COUNT = 56
EXPECTED_D8_CHECKS = 38
RETRACTION_PERCENTS = ("0", "10", "50", "90")
D6_PARITY_TOLERANCE = 1e-6
D6_HEADLINE_METHOD = "Chelated_ZeroMu_CSLS_8B"

NON_NDA_TARGETS: Tuple[str, ...] = (
    "docs/research/external-investor-teaser-non-nda.md",
    "docs/research/external-investor-teaser-deck-non-nda.html",
    "demo_workbench/compliance_copilot.html",
)

PART7_REQUIRED_PREFIXES: Tuple[str, ...] = ("wave11_d6_", "wave11_d8_")
PART7_OPTIONAL_PREFIXES: Tuple[str, ...] = ("wave11_d7_", "wave11_d9_", "wave11_d10_", "wave14_d11_")

# Wave 14b D11 PM6 takedown carry-forward bundle (verified whenever its directory exists).
D11_PREFIX = "wave14_d11_"
D11_MANIFEST_SCHEMA = "wave14_d11_pm6_manifest_v2"
D11_REANALYSIS_NAME = "pm6_posthoc_statistical_reanalysis.json"
D11_REANALYSIS_SCHEMA = "wave15_pm6_posthoc_statistical_reanalysis_v1"
D11_REANALYSIS_SCRIPT_REL = "scripts/compute_wave14b_pm6_posthoc_reanalysis.py"
D11_RECEIPT_SCHEMA = "copilot_receipt_v2"
D11_EXPECTED_RECEIPTS = 12
D11_RECEIPT_ALGORITHM = "Ed25519 over the ASCII hex of SHA-256(prev_hash || canonical_json(payload))"
D11_TARGET_BENIGN_FPR = 0.02
D11_HOLM_ALPHA = 0.05
D11_HEADLINE_ARM = "arm9_chelated_full_dual_gate"
D11_PRIMARY_COMPARISONS: Tuple[str, ...] = (
    "arm9_chelated_vs_arm3_tier2a_only",
    "arm9_chelated_vs_arm6_raw_cosine",
    "arm9_chelated_vs_arm8_linear_bridge",
)
D11_CROSS_LINGUAL_COMPARISONS: Tuple[str, ...] = (
    "paired_bootstrap_c_vs_a",
    "paired_bootstrap_c_vs_b",
    "paired_bootstrap_d_vs_c",
)
D11_PROCESS_GATES: Tuple[str, ...] = (
    "all_dev_calibrations_met_target_fpr",
    "bridge_effective_rank_384",
    "caa_sweep_completed_all_alphas",
    "phase0_min_yield_met",
    "unarmed_async_forced_bound_violations_zero",
    "zero_test_leakage_verified",
)
D11_DISCLOSURE_IDS: Tuple[str, ...] = tuple(f"D{i}" for i in range(1, 12))
D11_PRE_FREEZE_COMMITS: Tuple[str, ...] = ("f140e23b", "f6a82cd2", "64ec9fcf", "f357e4ba", "60ffc1f7")
D11_BRIDGE_RANK_FLOOR_ACTUALLY_CHECKED = 380

# Formal spec lock: Spec.lean is frozen before proofs; any edit must update this pin deliberately.
SPEC_LOCK_REL = "formal/spec_sha256.json"
EXPECTED_SPEC_SHA256 = "63fe3e644bdef93160c5d612bfc7d7086af03c3d15cf34417402a89a86cdc6eb"
CONFORMANCE_REPORT_REL = "formal/conformance_report.json"
EXPECTED_INCREMENTAL_DELETIONS_STATUS = "conformance_tested_unproven_in_lean"

RAW_SUFFIXES = (".csv", ".jsonl")
ANCILLARY_RECEIPT_KEYS = {
    "m2_preflight_sha256": "s1_m2_preflight.json",
    "selection_lock_sha256": "selection_lock.json",
}

D8_REPLAY_TIMEOUT_S = 120

# Executed in a subprocess with the target repo on sys.path; prints one JSON object.
D8_REPLAY_SOURCE = r"""
import hashlib, json, sys
repo = sys.argv[1]
sys.path.insert(0, repo)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from demo_workbench import compliance_copilot_engine as E
seed = hashlib.sha256(b"wave11-d8-runner-fixed-public-test-key").digest()
state = {"t": 1_700_000_000_000_000}
def tick():
    state["t"] += 1
    return state["t"]
svc = E.CopilotService(nda_token=None, clock_us=tick, private_key=Ed25519PrivateKey.from_private_bytes(seed))
entries = []
for sid, sc in E.SCENARIOS.items():
    for k, step in enumerate(sc.steps, 1):
        if step.retraction is None:
            continue
        r = svc.issue_receipt(sid, k, "non_nda")
        entries.append(r["entry_hash"] if isinstance(r, dict) and "entry_hash" in r else None)
ledger = svc.verify_ledger()
chain = [e["entry_hash"] for e in svc.ledger]
print(json.dumps({"ledger": ledger, "entry_hashes": chain, "public_key_b64": svc.public_key_b64,
                  "key_id": svc.key_id}))
"""


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
class VerifyError(Exception):
    """Raised for malformed inputs that make a check unable to complete."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VerifyError(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise VerifyError(f"invalid JSON in {path}: {exc}") from exc


def is_raw_gitignored(name: str) -> bool:
    return Path(name).suffix.lower() in RAW_SUFFIXES


def is_sealed_key(name: str) -> bool:
    return "sealed_key" in name.lower()


def safe_relative(name: str) -> bool:
    """True when ``name`` is a relative path that cannot escape its bundle directory."""
    p = Path(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts


class Report:
    """Ordered list of check results with PASS / FAIL / WARN / SKIP / NOT_RUN statuses."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str = "", **extra: Any) -> None:
        row: Dict[str, Any] = {"name": name, "status": status, "detail": detail}
        row.update(extra)
        self.checks.append(row)

    def passed(self, name: str, detail: str = "", **extra: Any) -> None:
        self.add(name, "PASS", detail, **extra)

    def failed(self, name: str, detail: str, **extra: Any) -> None:
        self.add(name, "FAIL", detail, **extra)

    def result(self, name: str, ok: bool, detail_ok: str, detail_bad: str, **extra: Any) -> None:
        if ok:
            self.passed(name, detail_ok, **extra)
        else:
            self.failed(name, detail_bad, **extra)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if c["status"] == "FAIL"]


# --------------------------------------------------------------------------------------
# (b) Receipt / manifest / sidecar verification
# --------------------------------------------------------------------------------------
def verify_receipt_dir(
    bdir: Path,
    receipt_name: str,
    manifest_name: Optional[str],
    ancillary_names: Sequence[str],
    label: str,
) -> Dict[str, Any]:
    """Verify one receipt-bound bundle directory; never raises for ordinary integrity faults."""
    errors: List[str] = []
    result: Dict[str, Any] = {
        "id": label,
        "dir": str(bdir),
        "errors": errors,
        "verified_side_artifacts": 0,
        "raw_hashed_present": 0,
        "raw_absent_optional": 0,
        "sealed_key_not_read": 0,
        "unhashed_historical_raw_files": 0,
        "manifest_sha256": None,
    }
    rpath = bdir / receipt_name
    if not rpath.is_file():
        errors.append(f"receipt missing: {receipt_name}")
        return result
    try:
        receipt = load_json(rpath)
    except VerifyError as exc:
        errors.append(str(exc))
        return result
    if not isinstance(receipt, dict):
        errors.append("receipt is not a JSON object")
        return result

    manifest_file = manifest_name or receipt.get("manifest_file")
    if not isinstance(manifest_file, str) or not safe_relative(manifest_file):
        errors.append("manifest file name unavailable or unsafe")
        return result
    mpath = bdir / manifest_file
    if not mpath.is_file():
        errors.append(f"manifest missing: {manifest_file}")
        return result

    actual = sha256_file(mpath)
    result["manifest_sha256"] = actual
    expected = receipt.get("manifest_sha256")
    if not isinstance(expected, str):
        errors.append("receipt has no manifest_sha256")
    elif actual != expected:
        errors.append(f"manifest sha256 mismatch: actual={actual[:16]} receipt={expected[:16]}")

    hashed_names = {manifest_file}

    for rkey, default_name in ANCILLARY_RECEIPT_KEYS.items():
        if rkey in receipt:
            fpath = bdir / default_name
            if not fpath.is_file():
                errors.append(f"ancillary missing: {default_name}")
            elif sha256_file(fpath) != str(receipt[rkey]):
                errors.append(f"ancillary hash mismatch: {default_name}")
            else:
                result["verified_side_artifacts"] += 1
            hashed_names.add(default_name)
    for aname in ancillary_names:
        if not safe_relative(aname) or not (bdir / aname).is_file():
            errors.append(f"spec ancillary JSON missing: {aname}")
            continue
        hashed_names.add(aname)

    for dict_key in ("artifact_sha256", "artifact_hashes"):
        side = receipt.get(dict_key)
        if not isinstance(side, dict):
            continue
        for fname, exp_sha in side.items():
            hashed_names.add(fname)
            if not safe_relative(fname):
                errors.append(f"unsafe sidecar path in {dict_key}: {fname!r}")
                continue
            fpath = bdir / fname
            if is_sealed_key(fname):
                # Sealed keys are never opened; only their presence is recorded.
                result["sealed_key_not_read"] += 1
                continue
            if not fpath.is_file():
                if is_raw_gitignored(fname):
                    result["raw_absent_optional"] += 1
                    continue
                errors.append(f"sidecar missing: {fname}")
                continue
            if sha256_file(fpath) != str(exp_sha):
                errors.append(f"sidecar sha256 mismatch: {fname}")
                continue
            if is_raw_gitignored(fname):
                result["raw_hashed_present"] += 1
            elif fname != manifest_file:
                result["verified_side_artifacts"] += 1

    for fpath in sorted(bdir.iterdir()):
        if fpath.is_file() and is_raw_gitignored(fpath.name) and fpath.name not in hashed_names:
            result["unhashed_historical_raw_files"] += 1
    return result


def verify_spec_bundles(repo_root: Path, spec: Dict[str, Any], report: Report) -> Dict[str, Any]:
    bundles = spec.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        report.failed("registry.spec_bundles", "spec has no bundles list")
        return {"total": 0, "verified": 0, "pending": 0, "failed": 0, "dirs": set()}
    verified = pending = failed = 0
    dirs = set()
    for entry in bundles:
        bid = str(entry.get("id", "?"))
        dirs.add(str(entry.get("dir", "")).replace("\\", "/").rstrip("/"))
        bdir = repo_root / str(entry.get("dir", ""))
        rpath = bdir / str(entry.get("receipt", ""))
        mpath = bdir / str(entry.get("manifest", ""))
        if (not rpath.is_file() or not mpath.is_file()) and entry.get("optional_until_run", False):
            pending += 1
            report.add(f"bundle.{bid}", "WARN", "PENDING_RUN (optional_until_run, artifacts absent)")
            continue
        res = verify_receipt_dir(
            bdir,
            str(entry.get("receipt", "")),
            str(entry.get("manifest", "")),
            list(entry.get("ancillary_json_hashes", [])),
            bid,
        )
        if res["errors"]:
            failed += 1
            report.failed(f"bundle.{bid}", "; ".join(res["errors"]))
        else:
            verified += 1
            report.passed(
                f"bundle.{bid}",
                f"manifest {str(res['manifest_sha256'])[:12]} ok; sidecars={res['verified_side_artifacts']} "
                f"raw_hashed={res['raw_hashed_present']} raw_absent_optional={res['raw_absent_optional']} "
                f"sealed_unread={res['sealed_key_not_read']}",
            )
    return {"total": len(bundles), "verified": verified, "pending": pending, "failed": failed, "dirs": dirs}


def discover_part7_dirs(repo_root: Path, spec_dirs: Iterable[str]) -> Tuple[List[Path], List[str]]:
    """Return (standalone Part VII dirs to verify, missing required prefixes)."""
    artifacts = repo_root / "artifacts"
    known = {d for d in spec_dirs}
    found: List[Path] = []
    missing: List[str] = []
    for prefix in PART7_REQUIRED_PREFIXES:
        matches = sorted(p for p in artifacts.glob(prefix + "*") if p.is_dir()) if artifacts.is_dir() else []
        in_spec = [m for m in matches if f"artifacts/{m.name}" in known]
        if not matches and not any(k.startswith(f"artifacts/{prefix}") for k in known):
            missing.append(prefix)
        for m in matches:
            if m not in in_spec:
                found.append(m)
    for prefix in PART7_OPTIONAL_PREFIXES:
        if not artifacts.is_dir():
            break
        for m in sorted(p for p in artifacts.glob(prefix + "*") if p.is_dir()):
            if f"artifacts/{m.name}" in known:
                continue
            if (m / "report_receipt.json").is_file():
                found.append(m)
    return found, missing


def verify_part7(repo_root: Path, spec_dirs: Iterable[str], report: Report) -> int:
    dirs, missing = discover_part7_dirs(repo_root, spec_dirs)
    for prefix in missing:
        report.failed(f"part7.{prefix.rstrip('_')}", f"required standalone bundle directory artifacts/{prefix}* missing")
    count = 0
    for d in dirs:
        res = verify_receipt_dir(d, "report_receipt.json", None, [], d.name)
        count += 1
        if res["errors"]:
            report.failed(f"part7.{d.name}", "; ".join(res["errors"]))
        else:
            report.passed(f"part7.{d.name}", f"manifest {str(res['manifest_sha256'])[:12]} ok")
    return count


def verify_pristine_mirror(repo_root: Path, spec: Dict[str, Any], report: Report) -> None:
    rel = spec.get("pristine_mirror_manifest")
    if not rel:
        report.add("pristine_mirror", "SKIP", "spec declares no pristine_mirror_manifest")
        return
    mpath = repo_root / str(rel)
    if not mpath.is_file():
        report.failed("pristine_mirror", f"manifest missing: {rel}")
        return
    try:
        data = load_json(mpath)
    except VerifyError as exc:
        report.failed("pristine_mirror", str(exc))
        return
    files = data.get("files", {})
    if isinstance(files, dict):
        items = [(k, v["sha256"] if isinstance(v, dict) else str(v)) for k, v in files.items()]
    else:
        items = [(str(e["path"]), str(e["sha256"])) for e in files]
    bad: List[str] = []
    for rel_file, exp in items:
        fp = mpath.parent / rel_file
        if not fp.is_file():
            bad.append(f"MISSING:{rel_file}")
        elif sha256_file(fp) != exp:
            bad.append(f"MISMATCH:{rel_file}")
    report.result(
        "pristine_mirror",
        not bad,
        f"{len(items)}/{len(items)} files byte-exact",
        f"{len(bad)} faults: {bad[:5]}",
    )


# --------------------------------------------------------------------------------------
# (c) D8 receipts: Ed25519 + hash chain + live replay
# --------------------------------------------------------------------------------------
def verify_d8_receipt_crypto(manifest: Dict[str, Any]) -> List[str]:
    """Pure-crypto verification of the sealed D8 ledger; returns a list of faults (empty = OK)."""
    faults: List[str] = []
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        return [f"cryptography package unavailable: {exc}"]

    det = manifest.get("deterministic")
    receipts_block = det.get("receipts") if isinstance(det, dict) else None
    if not isinstance(receipts_block, dict):
        return ["manifest.deterministic.receipts missing"]
    rows = receipts_block.get("receipts")
    key_block = receipts_block.get("signing_key")
    if not isinstance(rows, list) or not rows:
        return ["no receipt rows in manifest"]
    if not isinstance(key_block, dict) or key_block.get("algorithm") != "Ed25519":
        return ["signing_key block missing or not Ed25519"]
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(str(key_block.get("public_key_b64")), validate=True))
    except (ValueError, binascii.Error) as exc:
        return [f"public key undecodable: {exc}"]

    prev = GENESIS_HASH
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            faults.append(f"row {index} is not an object")
            continue
        if row.get("seq") != index:
            faults.append(f"row {index}: seq={row.get('seq')!r}")
        if row.get("prev_hash") != prev:
            faults.append(f"row {index}: prev_hash does not link to previous entry_hash")
        entry_hash = row.get("entry_hash")
        if not isinstance(entry_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", entry_hash):
            faults.append(f"row {index}: malformed entry_hash")
            continue
        try:
            sig = base64.b64decode(str(row.get("signature_b64")), validate=True)
            pub.verify(sig, entry_hash.encode("ascii"))
        except (ValueError, binascii.Error):
            faults.append(f"row {index}: signature undecodable")
        except InvalidSignature:
            faults.append(f"row {index}: Ed25519 signature INVALID")
        prev = entry_hash

    head = receipts_block.get("head_hash")
    if head != prev:
        faults.append("head_hash does not equal the last entry_hash")
    ledger = receipts_block.get("ledger")
    if not isinstance(ledger, dict) or ledger.get("intact") is not True or ledger.get("length") != len(rows) \
            or ledger.get("head_hash") != prev:
        faults.append("ledger summary inconsistent with receipt rows")
    return faults


def run_d8_replay(repo_root: Path, timeout_s: int = D8_REPLAY_TIMEOUT_S) -> Dict[str, Any]:
    """Re-issue the sealed receipts with the engine in a subprocess; raises VerifyError on failure."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", D8_REPLAY_SOURCE, str(repo_root)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerifyError(f"D8 replay could not run: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise VerifyError(f"D8 replay subprocess rc={proc.returncode}: {' | '.join(tail)}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise VerifyError(f"D8 replay produced unparsable output: {exc}") from exc


def compare_d8_replay(manifest: Dict[str, Any], replay: Dict[str, Any]) -> List[str]:
    faults: List[str] = []
    sealed = manifest["deterministic"]["receipts"]
    sealed_hashes = [r["entry_hash"] for r in sealed["receipts"]]
    if replay.get("entry_hashes") != sealed_hashes:
        faults.append("replayed entry_hashes differ from sealed manifest (engine or EK5 store drift)")
    ledger = replay.get("ledger", {})
    if not (isinstance(ledger, dict) and ledger.get("intact") is True and ledger.get("length") == len(sealed_hashes)):
        faults.append("engine verify_ledger() did not report an intact ledger of the sealed length")
    if ledger.get("head_hash") != sealed.get("head_hash"):
        faults.append("replayed head_hash differs from sealed head_hash")
    if replay.get("public_key_b64") != sealed["signing_key"].get("public_key_b64"):
        faults.append("replayed public key differs from sealed signing key")
    if replay.get("key_id") != sealed["signing_key"].get("key_id"):
        faults.append("replayed key_id differs from sealed key_id")
    return faults


def verify_d8(repo_root: Path, report: Report, replay_enabled: bool) -> Dict[str, Any]:
    info: Dict[str, Any] = {"replay": "SKIPPED_BY_FLAG" if not replay_enabled else "NOT_RUN"}
    ddir = None
    arts = repo_root / "artifacts"
    if arts.is_dir():
        cands = sorted(p for p in arts.glob("wave11_d8_*") if p.is_dir())
        ddir = cands[0] if cands else None
    if ddir is None or not (ddir / "report_receipt.json").is_file():
        report.failed("d8.receipts", "wave11_d8 bundle missing")
        return info
    try:
        receipt = load_json(ddir / "report_receipt.json")
        manifest = load_json(ddir / str(receipt.get("manifest_file", "")))
    except VerifyError as exc:
        report.failed("d8.receipts", str(exc))
        return info

    faults = verify_d8_receipt_crypto(manifest)
    n_rows = len(manifest.get("deterministic", {}).get("receipts", {}).get("receipts", []))
    report.result(
        "d8.ed25519_and_hash_chain",
        not faults,
        f"{n_rows} sealed receipts: Ed25519 signatures + SHA-256 chain + head verified",
        "; ".join(faults),
    )

    files = manifest.get("deterministic", {}).get("files_sha256", {})
    file_faults: List[str] = []
    html_rel = "demo_workbench/compliance_copilot.html"
    engine_key = "demo_workbench/compliance_copilot_engine.py"
    if isinstance(files, dict):
        if html_rel in files:
            hp = repo_root / html_rel
            if not hp.is_file() or sha256_file(hp) != files[html_rel]:
                file_faults.append(f"{html_rel} differs from sealed sha256")
        else:
            file_faults.append(f"{html_rel} not recorded in files_sha256")
        if engine_key in files:
            # The runner records engine_file_sha256(), which hashes ek5_v3_stratified.py (the lineage
            # store the engine depends on), under the engine's path. We verify that real binding.
            ek5 = repo_root / "ek5_v3_stratified.py"
            if not ek5.is_file() or sha256_file(ek5) != files[engine_key]:
                file_faults.append("ek5_v3_stratified.py differs from the sha256 sealed under the engine key")
        else:
            file_faults.append(f"{engine_key} not recorded in files_sha256")
    else:
        file_faults.append("files_sha256 missing")
    report.result(
        "d8.sealed_file_hashes",
        not file_faults,
        "page html and ek5_v3_stratified.py (sealed under the engine key) match the manifest",
        "; ".join(file_faults),
    )

    if not replay_enabled:
        report.add("d8.live_replay", "SKIP", "disabled by --no-d8-replay (reduced assurance, recorded in JSON)")
        return info
    try:
        replay = run_d8_replay(repo_root)
    except VerifyError as exc:
        report.failed("d8.live_replay", str(exc))
        info["replay"] = "FAIL"
        return info
    rfaults = compare_d8_replay(manifest, replay)
    info["replay"] = "FAIL" if rfaults else "PASS"
    report.result(
        "d8.live_replay",
        not rfaults,
        f"engine replay reproduced {len(replay.get('entry_hashes', []))} sealed entry hashes; "
        f"head {str(replay['ledger'].get('head_hash'))[:12]}",
        "; ".join(rfaults),
    )
    return info


# --------------------------------------------------------------------------------------
# (c2) D11 PM6 receipts: Ed25519 + payload-recomputed SHA-256 chain + Non-NDA rescan
# --------------------------------------------------------------------------------------
def _d11_canonical(payload: Any) -> bytes:
    """Integer-only canonical JSON matching ``chelated_takedown_firewall.canonical_json``."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _contains_float(obj: Any) -> bool:
    if isinstance(obj, float):
        return True
    if isinstance(obj, dict):
        return any(_contains_float(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_float(v) for v in obj)
    return False


def verify_d11_receipt_crypto(
    receipts_doc: Any,
    report_receipt: Dict[str, Any],
    manifest: Optional[Dict[str, Any]] = None,
    scanner: Any = None,
) -> List[str]:
    """Re-verify the sealed D11 ``copilot_receipt_v2`` ledger from first principles.

    Unlike D8, every entry carries its payload, so each ``entry_hash`` is recomputed as
    ``SHA-256(prev_hash || canonical_json(payload))`` instead of being trusted. Returns faults (empty = OK).
    """
    faults: List[str] = []
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        return [f"cryptography package unavailable: {exc}"]
    if not isinstance(receipts_doc, dict):
        return ["receipts document is not a JSON object"]
    if receipts_doc.get("schema_version") != D11_RECEIPT_SCHEMA:
        faults.append(f"schema_version={receipts_doc.get('schema_version')!r}")
    if receipts_doc.get("mode") != "non_nda":
        faults.append(f"mode={receipts_doc.get('mode')!r}, expected 'non_nda'")
    entries = receipts_doc.get("entries")
    if not isinstance(entries, list) or not entries:
        return faults + ["no receipt entries"]
    if len(entries) != D11_EXPECTED_RECEIPTS:
        faults.append(f"{len(entries)} entries, expected {D11_EXPECTED_RECEIPTS}")
    if receipts_doc.get("count") != len(entries):
        faults.append(f"count={receipts_doc.get('count')!r} != {len(entries)} entries")
    if report_receipt.get("receipts_entries_verified") != len(entries):
        faults.append("report_receipt.receipts_entries_verified does not equal the entry count")

    pub_b64 = str(receipts_doc.get("public_key_b64") or "")
    try:
        raw_pub = base64.b64decode(pub_b64, validate=True)
        pub = Ed25519PublicKey.from_public_bytes(raw_pub)
    except (ValueError, binascii.Error) as exc:
        return faults + [f"public key undecodable: {exc}"]
    key_id = hashlib.sha256(raw_pub).hexdigest()[:16]
    if receipts_doc.get("key_id") != key_id:
        faults.append("key_id is not sha256(public key)[:16]")

    prev = GENESIS_HASH
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
            faults.append(f"entry {index}: malformed")
            continue
        payload = entry["payload"]
        if entry.get("algorithm") != D11_RECEIPT_ALGORITHM:
            faults.append(f"entry {index}: algorithm={entry.get('algorithm')!r}")
        if _contains_float(payload):
            faults.append(f"entry {index}: float in integer-only payload")
        if payload.get("seq") != index:
            faults.append(f"entry {index}: payload.seq={payload.get('seq')!r}")
        if entry.get("prev_hash") != prev or payload.get("prev_hash") != prev:
            faults.append(f"entry {index}: prev_hash does not link to the previous entry_hash")
        if payload.get("key_id") != key_id or entry.get("public_key_b64") != pub_b64:
            faults.append(f"entry {index}: not signed under the ledger key")
        if payload.get("mode") != "non_nda":
            faults.append(f"entry {index}: payload.mode={payload.get('mode')!r}")
        entry_hash = entry.get("entry_hash")
        if not isinstance(entry_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", entry_hash):
            faults.append(f"entry {index}: malformed entry_hash")
            continue
        try:
            recomputed = sha256_bytes(str(entry.get("prev_hash")).encode("ascii") + _d11_canonical(payload))
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            recomputed = f"uncomputable: {exc}"
        if recomputed != entry_hash:
            faults.append(f"entry {index}: entry_hash does not recompute from prev_hash || canonical_json(payload)")
        try:
            pub.verify(base64.b64decode(str(entry.get("signature_b64")), validate=True), entry_hash.encode("ascii"))
        except (ValueError, binascii.Error):
            faults.append(f"entry {index}: signature undecodable")
        except InvalidSignature:
            faults.append(f"entry {index}: Ed25519 signature INVALID")
        prev = entry_hash

    chain = receipts_doc.get("chain_detail")
    if (
        not isinstance(chain, dict)
        or chain.get("intact") is not True
        or chain.get("length") != len(entries)
        or chain.get("head_hash") != prev
    ):
        faults.append("chain_detail inconsistent with the recomputed chain")
    if report_receipt.get("receipts_head_hash") != prev:
        faults.append("recomputed head_hash does not equal report_receipt.receipts_head_hash")
    if manifest is not None:
        ref = manifest.get("copilot_receipts_v2_ref")
        if not isinstance(ref, dict) or ref.get("head_hash") != prev or ref.get("entries_verified") != len(entries):
            faults.append("manifest.copilot_receipts_v2_ref does not bind the recomputed head / entry count")
    for flag in ("all_entries_valid", "chain_verified", "zero_floats_verified"):
        if receipts_doc.get(flag) is not True:
            faults.append(f"{flag} is not True")
    if receipts_doc.get("non_nda_leak_hits") != 0 or report_receipt.get("non_nda_leak_hits") != 0:
        faults.append("non_nda_leak_hits != 0")
    if scanner is not None:
        if report_receipt.get("non_nda_pattern_count") != len(scanner.FORBIDDEN_METHOD_PATTERNS):
            faults.append("report_receipt.non_nda_pattern_count differs from the live scanner")
        leaks = scanner.scan_text_for_ip_leaks(json.dumps(entries, sort_keys=True), source_label="d11_receipts")
        if leaks:
            faults.append(f"Non-NDA rescan of receipt entries found {len(leaks)} hits")
    return faults


def d11_holm(tests: Sequence[Tuple[str, float]], alpha: float = D11_HOLM_ALPHA) -> Dict[str, Tuple[bool, float]]:
    """Independent Holm-Bonferroni step-down: test_id -> (survives, adjusted_p)."""
    m = len(tests)
    out: Dict[str, Tuple[bool, float]] = {}
    running, rejecting = 0.0, True
    for rank, (tid, p) in enumerate(sorted(tests, key=lambda t: (t[1], t[0])), 1):
        mult = m - rank + 1
        running = max(running, min(1.0, mult * p))
        rejecting = rejecting and p <= alpha / mult
        out[tid] = (rejecting, running)
    return out


def d11_manifest_tests(manifest: Dict[str, Any]) -> List[Tuple[str, float]]:
    """The 12-test family (6 primary arm-AUC + 6 cross-lingual) read straight from the sealed manifest."""
    tests: List[Tuple[str, float]] = []
    for gen, comps in sorted(manifest.get("paired_arm_auc_comparisons_by_generator", {}).items()):
        for key in D11_PRIMARY_COMPARISONS:
            tests.append((f"{gen}:{key}", float(comps[key]["p_value_two_sided"])))
    langs = manifest.get("phase3_cross_lingual_4way_ablation", {}).get("languages", {})
    for lang in sorted(langs):
        for key in D11_CROSS_LINGUAL_COMPARISONS:
            tests.append((f"phase3:{lang}:{key}", float(langs[lang][key]["p_value_two_sided"])))
    return tests


def check_d11_headlines(manifest: Dict[str, Any], manifest_sha256: str, reanalysis: Optional[Dict[str, Any]]) -> List[str]:
    """Cross-check the sealed D11 manifest and its post-hoc addendum; every claim is recomputed here."""
    faults: List[str] = []
    if manifest.get("schema_version") != D11_MANIFEST_SCHEMA:
        faults.append(f"schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "COMPLETED" or manifest.get("seal_eligible") is not True:
        faults.append("manifest is not COMPLETED / seal_eligible")
    if manifest.get("execution_mode") != "run-live":
        faults.append(f"execution_mode={manifest.get('execution_mode')!r}")
    gates = manifest.get("gates")
    if not isinstance(gates, dict) or sorted(gates) != sorted(D11_PROCESS_GATES):
        faults.append("process gate set differs from the 6 sealed gates")
    elif not all(v is True for v in gates.values()) or manifest.get("all_gates_pass") is not True:
        faults.append("a sealed process gate is not True")
    gov = manifest.get("governance_header", {})
    if not isinstance(gov, dict) or gov.get("unlearning_claimed") is not False \
            or gov.get("parametric_recall_neutering_claimed") is not False:
        faults.append("governance header claims unlearning / parametric neutering")

    # Empirical TEST target r5_benign_fpr_le_002, recomputed from the headline arm's raw counts.
    r5_pass: Dict[str, bool] = {}
    for gen, arms in sorted(manifest.get("nine_arm_matrix_by_generator", {}).items()):
        fpr = arms.get(D11_HEADLINE_ARM, {}).get("r5_benign_fpr", {}) if isinstance(arms, dict) else {}
        try:
            k, n = int(fpr["false_refusals"]), int(fpr["n_families"])
        except (KeyError, TypeError, ValueError):
            faults.append(f"{gen}: headline-arm r5_benign_fpr counts missing")
            continue
        r5_pass[gen] = n > 0 and k / n <= D11_TARGET_BENIGN_FPR
    if not r5_pass:
        faults.append("no generator carries headline-arm R5 counts")

    if reanalysis is None:
        faults.append(f"{D11_REANALYSIS_NAME} missing: the post-hoc statistical/scope disclosure is mandatory")
        return faults
    if reanalysis.get("schema_version") != D11_REANALYSIS_SCHEMA:
        faults.append(f"reanalysis schema_version={reanalysis.get('schema_version')!r}")
    if reanalysis.get("source_manifest_sha256") != manifest_sha256:
        faults.append("reanalysis.source_manifest_sha256 does not bind the sealed manifest")
    if reanalysis.get("gpu_jobs_executed") != 0 or reanalysis.get("sealed_files_mutated") is not False:
        faults.append("reanalysis does not attest gpu_jobs_executed=0 / sealed_files_mutated=false")

    target = (
        reanalysis.get("process_vs_empirical_gates", {})
        .get("empirical_test_targets", {})
        .get("r5_benign_fpr_le_002", {})
    )
    by_gen = target.get("by_generator", {})
    for gen, ok in r5_pass.items():
        if by_gen.get(gen, {}).get("passed") is not ok:
            faults.append(f"reanalysis r5_benign_fpr_le_002[{gen}].passed misreports the recomputed value {ok}")
    all_ok = bool(r5_pass) and all(r5_pass.values())
    if target.get("passed_all_generators") is not all_ok:
        faults.append(f"reanalysis r5_benign_fpr_le_002.passed_all_generators != recomputed {all_ok}")
    if not all_ok and target.get("verdict") != "FAILED":
        faults.append("r5_benign_fpr_le_002 failed but the reanalysis does not disclose verdict FAILED")

    holm = reanalysis.get("holm_bonferroni_family", {}).get("full_family", {})
    try:
        expected = d11_holm(d11_manifest_tests(manifest))
    except (KeyError, TypeError, ValueError) as exc:
        faults.append(f"manifest Holm inputs unreadable: {exc}")
        expected = {}
    got = {t.get("test_id"): t for t in holm.get("tests", []) if isinstance(t, dict)}
    if not expected or set(got) != set(expected):
        faults.append(f"Holm family membership differs: {sorted(set(got) ^ set(expected))[:4]}")
    else:
        for tid, (survives, adj) in expected.items():
            row = got[tid]
            try:
                adj_got = float(row.get("holm_adjusted_p"))
            except (TypeError, ValueError):
                adj_got = -1.0
            if row.get("survives_holm") is not survives or abs(adj_got - adj) > 1e-6:
                faults.append(f"Holm step-down misreported for {tid}")
    if sorted(holm.get("survivors", [])) != sorted(t for t, (s, _) in expected.items() if s):
        faults.append("Holm survivor list differs from the recomputation")

    bridge = reanalysis.get("bridge_geometry_audit", {})
    live_rank = manifest.get("bridge_diagnostics", {}).get("effective_rank")
    if (
        bridge.get("effective_rank_threshold_actually_checked") != D11_BRIDGE_RANK_FLOOR_ACTUALLY_CHECKED
        or bridge.get("live_effective_rank") != live_rank
        or bridge.get("stiefel_orthogonality_vacuous") is not True
    ):
        faults.append("bridge_geometry_audit does not disclose the >=380 floor / live rank / vacuous orthogonality")

    disc = reanalysis.get("pre_freeze_abort_chronology_and_corpus_disclosures", {})
    if list(disc.get("disclosure_ids", [])) != list(D11_DISCLOSURE_IDS):
        faults.append("disclosures D1-D11 missing or out of order")
    if list(disc.get("chronology_commit_order", [])) != list(D11_PRE_FREEZE_COMMITS):
        faults.append("pre-freeze abort chronology commit order differs")
    if not isinstance(reanalysis.get("full_9arm_comparison"), dict):
        faults.append("reanalysis section full_9arm_comparison missing")
    return faults


def find_d11_dir(repo_root: Path) -> Optional[Path]:
    arts = repo_root / "artifacts"
    if not arts.is_dir():
        return None
    cands = sorted(p for p in arts.glob(D11_PREFIX + "*") if p.is_dir() and (p / "report_receipt.json").is_file())
    return cands[0] if cands else None


def verify_d11(repo_root: Path, report: Report) -> Dict[str, Any]:
    """Verify the D11 bundle's receipts, headline metrics and post-hoc addendum whenever it exists."""
    info: Dict[str, Any] = {"present": False}
    ddir = find_d11_dir(repo_root)
    if ddir is None:
        for name in ("d11.ed25519_and_hash_chain", "metrics.d11"):
            report.add(name, "SKIP", f"no artifacts/{D11_PREFIX}* bundle with report_receipt.json")
        return info
    info["present"] = True
    info["dir"] = ddir.name
    try:
        receipt = load_json(ddir / "report_receipt.json")
        mname = str(receipt.get("manifest_file", ""))
        if not mname or not safe_relative(mname):
            raise VerifyError(f"unsafe or missing manifest_file {mname!r}")
        mpath = ddir / mname
        manifest = load_json(mpath)
        manifest_sha = sha256_file(mpath)
        if not isinstance(manifest, dict):
            raise VerifyError("D11 manifest is not a JSON object")
    except (VerifyError, AttributeError) as exc:
        report.failed("d11.ed25519_and_hash_chain", str(exc))
        report.failed("metrics.d11", str(exc))
        return info

    ref = manifest.get("copilot_receipts_v2_ref")
    rname = str(ref.get("file", "")) if isinstance(ref, dict) else ""
    try:
        scanner: Any = load_scanner(repo_root)
    except VerifyError:
        scanner = None
    if not rname or not safe_relative(rname) or not (ddir / rname).is_file():
        report.failed("d11.ed25519_and_hash_chain", "manifest.copilot_receipts_v2_ref.file missing or absent on disk")
    else:
        try:
            faults = verify_d11_receipt_crypto(load_json(ddir / rname), receipt, manifest, scanner)
        except VerifyError as exc:
            faults = [str(exc)]
        if scanner is None:
            faults.append("Non-NDA scanner unavailable for the receipt rescan")
        report.result(
            "d11.ed25519_and_hash_chain",
            not faults,
            f"{D11_EXPECTED_RECEIPTS} receipts: payload-recomputed SHA-256 chain + Ed25519 signatures; head == "
            "report_receipt.receipts_head_hash; zero floats; 0 Non-NDA hits on rescan",
            "; ".join(faults[:6]),
        )

    rpath = ddir / D11_REANALYSIS_NAME
    try:
        reanalysis = load_json(rpath) if rpath.is_file() else None
    except VerifyError as exc:
        report.failed("metrics.d11", str(exc))
        return info
    if reanalysis is not None and not isinstance(reanalysis, dict):
        report.failed("metrics.d11", f"{D11_REANALYSIS_NAME} is not a JSON object")
        return info
    faults = check_d11_headlines(manifest, manifest_sha, reanalysis)
    report.result(
        "metrics.d11",
        not faults,
        "6 sealed gates are process-only; r5_benign_fpr_le_002 recomputed and honestly disclosed FAILED; "
        "12-test Holm step-down recomputed; D1-D11 disclosures bound to the manifest sha256",
        "; ".join(faults[:6]),
    )

    script = repo_root / D11_REANALYSIS_SCRIPT_REL
    if script.is_file() and reanalysis is not None:
        ok, bad = False, ""
        try:
            spec = importlib.util.spec_from_file_location("_d11_posthoc_under_verify", script)
            if spec is None or spec.loader is None:
                raise VerifyError("cannot build import spec for the reanalysis script")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ok = module.render(module.build_reanalysis(ddir)) == rpath.read_bytes()
            bad = f"{D11_REANALYSIS_NAME} is not byte-identical to a fresh CPU recomputation"
        except Exception as exc:  # noqa: BLE001 - any recompute failure is a reported FAIL, never swallowed
            bad = f"recompute raised {type(exc).__name__}: {exc}"
        report.result(
            "d11.reanalysis_recompute",
            ok,
            f"{D11_REANALYSIS_NAME} byte-identical to a fresh recomputation from the sealed manifest",
            bad,
        )
    return info


# --------------------------------------------------------------------------------------
# (c3) Formal spec lock
# --------------------------------------------------------------------------------------
def check_spec_lock(repo_root: Path) -> Optional[List[str]]:
    """Faults for the formal spec lock, or None when ``formal/spec_sha256.json`` is absent."""
    lock_path = repo_root / SPEC_LOCK_REL
    if not lock_path.is_file():
        return None
    try:
        lock = load_json(lock_path)
    except VerifyError as exc:
        return [str(exc)]
    if not isinstance(lock, dict):
        return [f"{SPEC_LOCK_REL} is not a JSON object"]
    faults: List[str] = []
    rel = str(lock.get("spec_relpath") or "")
    claimed = str(lock.get("spec_sha256") or "")
    if claimed != EXPECTED_SPEC_SHA256:
        faults.append(f"spec_sha256.json claims {claimed[:12]!r}, pinned lock is {EXPECTED_SPEC_SHA256[:12]}")
    spec_path = repo_root / rel
    if not rel or not safe_relative(rel) or not spec_path.is_file():
        faults.append(f"locked spec file missing: {rel!r}")
    else:
        actual = sha256_file(spec_path)
        if actual != EXPECTED_SPEC_SHA256:
            faults.append(f"sha256({rel})={actual[:12]} != pinned {EXPECTED_SPEC_SHA256[:12]}")
    if lock.get("locked_before_proofs") is not True:
        faults.append("locked_before_proofs is not True")
    conf_path = repo_root / CONFORMANCE_REPORT_REL
    if not conf_path.is_file():
        faults.append(f"{CONFORMANCE_REPORT_REL} missing")
    else:
        try:
            conf = load_json(conf_path)
            status = conf.get("incremental_with_deletions_status") if isinstance(conf, dict) else None
        except VerifyError as exc:
            status = f"unreadable: {exc}"
        if status != EXPECTED_INCREMENTAL_DELETIONS_STATUS:
            faults.append(
                f"incremental_with_deletions_status={status!r}, expected {EXPECTED_INCREMENTAL_DELETIONS_STATUS!r}"
            )
    return faults


def verify_spec_lock(repo_root: Path, report: Report) -> None:
    faults = check_spec_lock(repo_root)
    if faults is None:
        report.add("formal.spec_sha256_lock", "SKIP", f"{SPEC_LOCK_REL} not present")
        return
    report.result(
        "formal.spec_sha256_lock",
        not faults,
        f"Spec.lean sha256 {EXPECTED_SPEC_SHA256[:12]} matches the pinned lock; incremental-with-deletions "
        "disclosed as conformance-tested, unproven in Lean",
        "; ".join(faults),
    )


# --------------------------------------------------------------------------------------
# (d) Non-NDA scan
# --------------------------------------------------------------------------------------
def load_scanner(repo_root: Path) -> Any:
    path = repo_root / "scripts" / "verify_non_nda_ip_clean.py"
    if not path.is_file():
        raise VerifyError(f"scanner missing: {path}")
    spec = importlib.util.spec_from_file_location("_verify_non_nda_ip_clean_under_test", path)
    if spec is None or spec.loader is None:
        raise VerifyError("cannot build import spec for scanner")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (OSError, SyntaxError, ImportError) as exc:
        raise VerifyError(f"scanner failed to import: {exc}") from exc
    return module


def verify_non_nda(repo_root: Path, report: Report) -> None:
    try:
        scanner = load_scanner(repo_root)
    except VerifyError as exc:
        report.failed("non_nda.scanner", str(exc))
        return
    n_patterns = len(scanner.FORBIDDEN_METHOD_PATTERNS)
    report.result(
        "non_nda.pattern_count",
        n_patterns == EXPECTED_PATTERN_COUNT,
        f"{n_patterns} patterns",
        f"scanner has {n_patterns} patterns, expected {EXPECTED_PATTERN_COUNT}",
    )
    for rel in NON_NDA_TARGETS:
        path = repo_root / rel
        if not path.is_file():
            report.failed(f"non_nda.{Path(rel).name}", f"file missing: {rel}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            report.failed(f"non_nda.{Path(rel).name}", f"unreadable: {exc}")
            continue
        leaks = scanner.scan_text_for_ip_leaks(text, source_label=path.name)
        report.result(
            f"non_nda.{Path(rel).name}",
            not leaks,
            f"0/{n_patterns} forbidden-pattern hits",
            f"{len(leaks)} hits: {leaks[:3]}",
        )


# --------------------------------------------------------------------------------------
# (e) Headline metrics + registry integrity
# --------------------------------------------------------------------------------------
def iter_d6_cases(manifest: Dict[str, Any]) -> Iterable[Tuple[str, str, str, str, str, Dict[str, Any]]]:
    for ds, per_dim in manifest.get("benchmarks", {}).items():
        for dim, block in per_dim.items():
            for engine, ecase in block.get("cases", {}).items():
                for direction, dblock in ecase.get("directions", {}).items():
                    for pct, cell in dblock.get("retraction_percent", {}).items():
                        yield ds, dim, engine, direction, pct, cell


def check_d6_headlines(manifest: Dict[str, Any]) -> List[str]:
    faults: List[str] = []
    if manifest.get("status") != "COMPLETED":
        faults.append(f"status={manifest.get('status')!r}")
    if manifest.get("failed_gates") != []:
        faults.append(f"failed_gates={manifest.get('failed_gates')!r}")
    pcts = set()
    groups: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    n_cells = 0
    for ds, dim, engine, direction, pct, cell in iter_d6_cases(manifest):
        n_cells += 1
        pcts.add(pct)
        rate = cell.get("stale_resurrection_rate")
        if rate != 0:
            faults.append(f"stale_resurrection_rate={rate!r} at {ds}/{dim}/{engine}/{direction}/{pct}%")
        quality = cell.get("quality")
        if quality:
            try:
                value = float(quality["ndcg_at_10"]["means"][D6_HEADLINE_METHOD])
            except (KeyError, TypeError, ValueError):
                faults.append(f"missing {D6_HEADLINE_METHOD} nDCG@10 at {ds}/{dim}/{engine}/{direction}/{pct}%")
                continue
            groups.setdefault((ds, dim, direction, pct), {})[engine] = value
    if n_cells == 0:
        faults.append("no D6 benchmark cells found")
    missing = [p for p in RETRACTION_PERCENTS if p not in pcts]
    if missing:
        faults.append(f"retraction percents not covered: {missing}")
    multi = [g for g in groups.values() if len(g) >= 2]
    if not multi:
        faults.append("no (dataset,dim,direction,pct) group has nDCG@10 from two or more engines")
    for key, vals in groups.items():
        if len(vals) >= 2 and max(vals.values()) - min(vals.values()) > D6_PARITY_TOLERANCE:
            faults.append(f"nDCG@10 parity broken at {key}: {vals}")
    return faults


def check_d8_headlines(manifest: Dict[str, Any]) -> List[str]:
    faults: List[str] = []
    checks = manifest.get("checks")
    if manifest.get("all_checks_pass") is not True:
        faults.append("all_checks_pass is not True")
    if manifest.get("checks_total") != EXPECTED_D8_CHECKS:
        faults.append(f"checks_total={manifest.get('checks_total')!r}, expected {EXPECTED_D8_CHECKS}")
    if manifest.get("checks_failed") != []:
        faults.append(f"checks_failed={manifest.get('checks_failed')!r}")
    if not isinstance(checks, list) or len(checks) != EXPECTED_D8_CHECKS or not all(
        isinstance(c, dict) and c.get("pass") is True for c in checks
    ):
        faults.append(f"not all {EXPECTED_D8_CHECKS} individual checks pass")
    det = manifest.get("deterministic", {})
    if det.get("forbidden_pattern_count") != EXPECTED_PATTERN_COUNT:
        faults.append(f"forbidden_pattern_count={det.get('forbidden_pattern_count')!r}")
    if isinstance(checks, list):
        hit_checks = [c for c in checks if "hit" in str(c.get("check", "")).lower()]
        if not hit_checks or any(c.get("detail") != [] or c.get("pass") is not True for c in hit_checks):
            faults.append("Non-NDA hit checks are missing or report hits")
    return faults


def check_d7_headlines(manifest: Dict[str, Any]) -> List[str]:
    faults: List[str] = []
    raw_txt = json.dumps(manifest, sort_keys=True)
    if "NEO4J_AUTH=neo4j/<REDACTED>" not in raw_txt:
        faults.append("NEO4J_AUTH=<REDACTED> missing in D7 manifest")
    if manifest.get("suite_a_n") != 60 or manifest.get("suite_b_n") != 20:
        faults.append(f"unexpected suite sizes ({manifest.get('suite_a_n')}, {manifest.get('suite_b_n')})")
    if manifest.get("suite_b_pooled") is not False:
        faults.append("suite_b_pooled must be False")
    sb = manifest.get("suite_b_descriptive") or {}
    if sb.get("inferential_claim") is not False:
        faults.append("suite_b_descriptive.inferential_claim must be False")
    settled = (manifest.get("suite_a_inference") or {}).get("settled") or {}
    chel_acc = float(((settled.get("arms") or {}).get("Chelated_v4_Stratified_WFS") or {}).get("accuracy", 0.0))
    if chel_acc < 0.90:
        faults.append(f"D7 settled Chelated accuracy={chel_acc}")
    holm = settled.get("holm_bonferroni") or {}
    if len(holm) != 2 or not all(bool((v or {}).get("reject_null_at_alpha")) for v in holm.values()):
        faults.append("D7 settled Holm-Bonferroni family did not reject null on both arms")
    return faults


def check_d9_headlines(manifest: Dict[str, Any]) -> List[str]:
    faults: List[str] = []
    pa = ((manifest.get("part_a_fast_edge_extractor") or {}).get("suite_a_evaluation_n60") or {})
    if float(pa.get("semantic_accuracy", 0.0)) < 0.95:
        faults.append(f"D9 Part A semantic_accuracy={pa.get('semantic_accuracy')}")
    l4 = pa.get("l4_gates") or {}
    if l4.get("ms_per_case_below_1500") is not False:
        faults.append("D9 Part A honest latency gate miss (ms_per_case_below_1500=False) missing")
    pb_sum = manifest.get("part_b_hypothesis_summary") or {}
    if pb_sum.get("verdict") != "K4 stands under Protocol v2":
        faults.append(f"D9 Part B verdict={pb_sum.get('verdict')!r}")
    return faults


def check_d10_headlines(manifest: Dict[str, Any]) -> List[str]:
    faults: List[str] = []
    pa = manifest.get("part_a_real_corpus_cutover") or {}
    if pa.get("total_real_documents_merged") != 9_465_539:
        faults.append(f"D10 total_real_documents_merged={pa.get('total_real_documents_merged')}")
    if float(pa.get("synthetic_vector_fraction", 1.0)) != 0.0:
        faults.append(f"D10 synthetic_vector_fraction={pa.get('synthetic_vector_fraction')}")
    fence = pa.get("retraction_fence_9_47m_proof") or {}
    if float(fence.get("chelated_fenced_stale_resurrection_rate", 1.0)) != 0.0:
        faults.append("D10 fenced stale resurrection rate != 0.0")
    mult = manifest.get("part_a_multiplicity_and_loss_disclosure") or {}
    if mult.get("n_contrasts_in_family") != 24:
        faults.append(f"D10 n_contrasts_in_family={mult.get('n_contrasts_in_family')!r} != 24")
    losses = mult.get("disclosed_chelated_point_or_ci_losses") or []
    if not any(bool(row.get("statistically_significant_loss")) for row in losses if isinstance(row, dict)):
        faults.append("D10 loss disclosure missing statistically_significant_loss entry (Quora vs RRF)")
    pb = manifest.get("part_b_70b_frontier_scale_sweep") or {}
    if pb.get("status") not in ("COMPLETED", "NOT_RUN"):
        faults.append(f"D10 Part B status={pb.get('status')!r}")
    return faults


def verify_headline_metrics(repo_root: Path, report: Report) -> None:
    arts = repo_root / "artifacts"
    for tag, prefix, checker, required, ok_msg in (
        ("d6", "wave11_d6_", check_d6_headlines, True, "cross-engine nDCG@10 parity + 0.00% stale resurrection at 0/10/50/90"),
        ("d8", "wave11_d8_", check_d8_headlines, True, f"{EXPECTED_D8_CHECKS}/{EXPECTED_D8_CHECKS} checks, 0/{EXPECTED_PATTERN_COUNT} Non-NDA hits"),
        ("d7", "wave11_d7_", check_d7_headlines, False, "redacted Neo4j auth, settled Suite A Holm reject + descriptive Suite B"),
        ("d9", "wave11_d9_", check_d9_headlines, False, "0.5B FastEdge 96.67% + latency gate miss + Protocol v2 K4 null disclosed"),
        ("d10", "wave11_d10_", check_d10_headlines, False, "9.47M real cutover (0% synthetic/stale, 24-contrast Holm + Quora loss) + Part B status"),
    ):
        cands = sorted(p for p in arts.glob(prefix + "*") if p.is_dir()) if arts.is_dir() else []
        if not cands or not (cands[0] / "report_receipt.json").is_file():
            if required:
                report.failed(f"metrics.{tag}", f"{prefix}* bundle directory missing")
            continue
        try:
            receipt = load_json(cands[0] / "report_receipt.json")
            manifest = load_json(cands[0] / str(receipt.get("manifest_file", "")))
        except VerifyError as exc:
            report.failed(f"metrics.{tag}", str(exc))
            continue
        faults = checker(manifest)
        report.result(f"metrics.{tag}", not faults, ok_msg, "; ".join(faults[:6]))


def registry_content_digest(registry: Dict[str, Any]) -> str:
    payload = {k: v for k, v in registry.items() if k != "registry_content_sha256"}
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def verify_registry_file(repo_root: Path, spec: Dict[str, Any], report: Report) -> None:
    path = repo_root / REGISTRY_REL
    if not path.is_file():
        report.add("registry.verified_registry", "SKIP", "verified_registry.json not present")
        return
    try:
        registry = load_json(path)
    except VerifyError as exc:
        report.failed("registry.verified_registry", str(exc))
        return
    claimed = registry.get("registry_content_sha256")
    actual = registry_content_digest(registry)
    report.result(
        "registry.content_sha256",
        claimed == actual,
        f"registry_content_sha256 {actual[:12]} recomputes",
        f"claimed {str(claimed)[:12]} != recomputed {actual[:12]}",
    )
    spec_ids = [str(b.get("id")) for b in spec.get("bundles", [])]
    reg_ids = [str(b.get("id")) for b in registry.get("bundles", [])]
    if spec_ids != reg_ids:
        report.add(
            "registry.matches_spec",
            "WARN",
            f"registry lists {len(reg_ids)} bundles but spec lists {len(spec_ids)}; per-bundle cross-check "
            "skipped (rebuild with scripts/build_whitepaper_assets.py)",
        )
        return
    drift: List[str] = []
    for rb in registry.get("bundles", []):
        if not rb.get("present"):
            continue
        entry = next((b for b in spec.get("bundles", []) if b.get("id") == rb.get("id")), None)
        if entry is None:
            continue
        mpath = repo_root / str(entry["dir"]) / str(entry["manifest"])
        if not mpath.is_file() or sha256_file(mpath) != rb.get("manifest_sha256"):
            drift.append(str(rb.get("id")))
    report.result(
        "registry.matches_spec",
        not drift,
        f"registry bundle manifest hashes match the on-disk manifests ({len(reg_ids)} bundles)",
        f"manifest drift vs registry: {drift[:5]}",
    )


# --------------------------------------------------------------------------------------
# (f) release_status honesty gate
# --------------------------------------------------------------------------------------
def evaluate_release_status(status: Any, pm4b_block: Any) -> List[str]:
    """Return faults; PRE-RELEASE must be retained while PM-4b human grading is pending."""
    faults: List[str] = []
    text = str(status or "")
    complete = isinstance(pm4b_block, dict) and str(pm4b_block.get("status", "")).upper() == "COMPLETE"
    if complete:
        evidence = pm4b_block.get("human_grader_receipt")
        if not evidence:
            faults.append("PM-4b claimed COMPLETE without a human_grader_receipt reference")
    elif "PRE-RELEASE" not in text.upper():
        faults.append(f"release_status {text!r} dropped PRE-RELEASE while PM-4b human grading is pending")
    if re.search(r"ship[- ]ready", text, re.IGNORECASE):
        faults.append("release_status contains forbidden promotional phrase 'Ship-Ready'")
    return faults


def verify_release_status(repo_root: Path, spec: Dict[str, Any], report: Report) -> None:
    faults = evaluate_release_status(spec.get("release_status"), spec.get("pm4b_human_grading"))
    reg_path = repo_root / REGISTRY_REL
    if reg_path.is_file():
        try:
            reg_status = load_json(reg_path).get("release_status")
        except VerifyError as exc:
            faults.append(str(exc))
        else:
            if reg_status != spec.get("release_status"):
                faults.append("verified_registry.json release_status differs from the spec")
    wp = repo_root / "docs/research/doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md"
    html = repo_root / "docs/research/chelated-memory-spherical-migration-whitepaper-presentation.html"
    for doc in (wp, html):
        if doc.is_file():
            body = doc.read_text(encoding="utf-8")
            if "PRE-RELEASE" not in body and "pm4b_human_grading" not in spec:
                faults.append(f"{doc.name} lacks the PRE-RELEASE banner")
            if re.search(r"\bShip-Ready\b", body, re.IGNORECASE):
                faults.append(f"{doc.name} contains 'Ship-Ready'")
    report.result(
        "release_status.honesty",
        not faults,
        f"{spec.get('release_status')!r} retained (PM-4b human grading pending)",
        "; ".join(faults),
    )


# --------------------------------------------------------------------------------------
# Formal tier (--full-formal)
# --------------------------------------------------------------------------------------
_AX_DEPENDS = re.compile(r"'([^']+)'\s+depends on axioms:\s*\[([^\]]*)\]", re.S)
_AX_NONE = re.compile(r"'([^']+)'\s+does not depend on any axioms")


def parse_axiom_report(text: str) -> Dict[str, List[str]]:
    """Parse Lean ``#print axioms`` output into ``{declaration: [axiom, ...]}``."""
    result: Dict[str, List[str]] = {}
    for m in _AX_DEPENDS.finditer(text):
        result[m.group(1)] = [a.strip() for a in m.group(2).replace("\n", " ").split(",") if a.strip()]
    for m in _AX_NONE.finditer(text):
        result.setdefault(m.group(1), [])
    return result


def evaluate_axiom_report(text: str, expected: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Apply the {propext, Classical.choice, Quot.sound} allowlist; rejects sorryAx / ofReduceBool / customs."""
    decls = parse_axiom_report(text)
    violations: List[str] = []
    errors: List[str] = []
    for line in text.splitlines():
        if re.search(r":\s*error\b|^error\b", line):
            errors.append(line.strip())
    for decl, axioms in sorted(decls.items()):
        bad = sorted(set(axioms) - ALLOWED_AXIOMS)
        if bad:
            violations.append(f"{decl}: {', '.join(bad)}")
    if expected is not None:
        missing = sorted(set(expected) - set(decls))
        if missing:
            errors.append(f"no axiom report for: {', '.join(missing[:10])}")
    if not decls:
        errors.append("no '#print axioms' output parsed (vacuous audit)")
    return {"ok": not violations and not errors, "declarations": len(decls), "violations": violations, "errors": errors}


def strip_lean_comments(text: str) -> str:
    """Remove ``--`` line comments, nested ``/- -/`` blocks and string literals, keeping line count."""
    out: List[str] = []
    i, n, depth = 0, len(text), 0
    while i < n:
        two = text[i:i + 2]
        if depth == 0 and two == "--":
            while i < n and text[i] != "\n":
                i += 1
        elif two == "/-":
            depth += 1
            i += 2
        elif depth > 0 and two == "-/":
            depth -= 1
            i += 2
        elif depth > 0:
            out.append("\n" if text[i] == "\n" else " ")
            i += 1
        elif text[i] == '"':
            out.append('"')
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    i += 1
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append('"')
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def scan_lean_forbidden(text: str) -> List[Tuple[int, str]]:
    """Return ``(line, token)`` for each ``sorry`` / ``native_decide`` outside comments and strings."""
    hits: List[Tuple[int, str]] = []
    for lineno, line in enumerate(strip_lean_comments(text).splitlines(), 1):
        for m in re.finditer(r"\b(sorry|native_decide)\b", line):
            hits.append((lineno, m.group(1)))
    return hits


_DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*((?:(?:private|protected|noncomputable|unsafe|partial)\s+)*)(theorem|lemma)\s+([^\s:({\[]+)"
)


def collect_lean_theorems(text: str) -> List[str]:
    """Fully-qualified public theorem / lemma names, tracking ``namespace`` / ``section`` / ``end``."""
    stack: List[Tuple[str, str]] = []
    names: List[str] = []
    for line in strip_lean_comments(text).splitlines():
        s = line.strip()
        m_ns = re.match(r"namespace\s+(\S+)", s)
        if m_ns:
            stack.append(("ns", m_ns.group(1)))
            continue
        if re.match(r"section\b", s):
            m_sec = re.match(r"section\s+(\S+)", s)
            stack.append(("sec", m_sec.group(1) if m_sec else ""))
            continue
        m_end = re.match(r"end(?:\s+(\S+))?\s*$", s)
        if m_end:
            if stack:
                stack.pop()
            continue
        m = _DECL_RE.match(line)
        if not m or "private" in m.group(1):
            continue
        name = m.group(3)
        if name.startswith("_root_."):
            names.append(name[len("_root_."):])
            continue
        prefix = ".".join(v for k, v in stack if k == "ns")
        names.append(f"{prefix}.{name}" if prefix else name)
    return names


def list_lean_files(formal_dir: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(formal_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in (".lake", "lake-packages", "build", ".git"))
        for fn in sorted(filenames):
            if fn.endswith(".lean") and not fn.startswith("lakefile"):
                files.append(Path(dirpath) / fn)
    return files


NANODA_OK_RE = re.compile(r"Checked\s+(\d+)\s+declarations?\s+with\s+no\s+typechecker\s+errors")
NANODA_PP_RE = re.compile(r"(\d+)\s+pretty[- ]printer\s+errors?", re.I)


def _exe_name(name: str) -> str:
    return name + ".exe" if os.name == "nt" else name


def _first_file(candidates: Sequence[Optional[str]]) -> Optional[str]:
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(cand)
    return None


def resolve_nanoda_bin(name: str = "nanoda_bin") -> Optional[str]:
    """``$NANODA_BIN`` -> ``PATH`` -> ``~/.cargo/nanoda_target/release`` -> ``~/.cargo/bin``; ``None`` if unresolved."""
    exe = _exe_name("nanoda_bin")
    return _first_file(
        [
            os.environ.get("NANODA_BIN"),
            shutil.which(name),
            str(Path.home() / ".cargo" / "nanoda_target" / "release" / exe),
            str(Path.home() / ".cargo" / "bin" / exe),
        ]
    )


def resolve_lean4export_bin(name: str = "lean4export") -> Optional[str]:
    """``$LEAN4EXPORT_BIN`` -> ``PATH`` -> ``~/.elan/bin`` -> ``~/.cargo/bin``; ``None`` if unresolved."""
    exe = _exe_name("lean4export")
    return _first_file(
        [
            os.environ.get("LEAN4EXPORT_BIN"),
            shutil.which(name),
            str(Path.home() / ".elan" / "bin" / exe),
            str(Path.home() / ".cargo" / "bin" / exe),
        ]
    )


def resolve_lake_bin(name: str = "lake") -> Optional[str]:
    """``$LAKE_BIN`` -> ``PATH`` -> ``~/.elan/bin``; ``None`` if unresolved."""
    env_lake = os.environ.get("LAKE_BIN")
    if env_lake and Path(env_lake).is_file():
        return str(env_lake)
    found = shutil.which(name)
    if found:
        return found
    elan_cand = Path.home() / ".elan" / "bin" / _exe_name("lake")
    if elan_cand.is_file():
        return str(elan_cand)
    return None


def parse_nanoda_output(rc: int, out: str, expected_declarations: int) -> Dict[str, Any]:
    """Accept only rc == 0 AND ``Checked N declarations with no typechecker errors`` with N > 0 and
    N >= ``expected_declarations``.  A bare rc 0 with no success line is rejected."""
    ok = NANODA_OK_RE.search(out)
    pp = NANODA_PP_RE.search(out)
    checked = int(ok.group(1)) if ok else None
    accepted = rc == 0 and checked is not None and checked > 0 and checked >= expected_declarations
    return {
        "accepted": accepted,
        "declarations_checked": checked,
        "pretty_printer_errors": int(pp.group(1)) if pp else 0,
    }


def load_nanoda_manifest(formal: Path) -> Optional[Dict[str, Any]]:
    path = formal / "nanoda_manifest.json"
    if not path.is_file():
        return None
    try:
        data = load_json(path)
    except VerifyError:
        return None
    return data if isinstance(data, dict) else None


def export_is_stale(formal: Path, manifest: Optional[Dict[str, Any]]) -> bool:
    """True when the export file is missing or older than any formal/**/*.lean or nanoda_manifest.json."""
    rel = str((manifest or {}).get("export_file") or ".lake/nanoda/chelated_formal_all.export")
    export = formal / rel
    if not export.is_file():
        return True
    export_mtime = export.stat().st_mtime
    sources = list_lean_files(formal)
    manifest_path = formal / "nanoda_manifest.json"
    if manifest_path.is_file():
        sources.append(manifest_path)
    return any(src.stat().st_mtime > export_mtime for src in sources)


def formal_prerequisites(repo_root: Path) -> List[str]:
    """Return reasons the formal tier cannot run (empty list = runnable)."""
    missing: List[str] = []
    formal = repo_root / "formal"
    if not formal.is_dir():
        return ["formal/ directory absent"]
    lakefile = formal / "lakefile.lean"
    lakefile_toml = formal / "lakefile.toml"
    if not lakefile.is_file() and not lakefile_toml.is_file():
        missing.append("formal/lakefile.lean or lakefile.toml absent")
    if resolve_lake_bin() is None:
        missing.append("`lake` not on PATH or in ~/.elan/bin")
    lake_text = ""
    for lf in (lakefile, lakefile_toml):
        if lf.is_file():
            lake_text += lf.read_text(encoding="utf-8", errors="replace")
    if "mathlib" in lake_text.lower():
        candidates = (
            formal / ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean",
            formal / ".lake/packages/mathlib/.lake/build/lib/Mathlib.olean",
        )
        if not any(c.is_file() for c in candidates):
            missing.append("Mathlib .olean cache absent under formal/.lake/packages/mathlib")
    manifest = load_nanoda_manifest(formal)
    if manifest is None:
        missing.append("formal/nanoda_manifest.json absent or unreadable (external kernel unconfigured)")
    else:
        decls = manifest.get("export_declarations")
        if not manifest.get("nanoda_args") or not isinstance(decls, list) or not decls:
            missing.append("formal/nanoda_manifest.json lacks nanoda_args / export_declarations (external kernel unconfigured)")
        if resolve_nanoda_bin(str(manifest.get("nanoda_bin") or "nanoda_bin")) is None:
            missing.append("nanoda_bin not found ($NANODA_BIN, PATH, ~/.cargo/nanoda_target/release, ~/.cargo/bin)")
        if export_is_stale(formal, manifest) and resolve_lean4export_bin(str(manifest.get("lean4export_bin") or "lean4export")) is None:
            missing.append(
                "lean4export file missing/stale and lean4export not found ($LEAN4EXPORT_BIN, PATH, ~/.elan/bin, ~/.cargo/bin)"
            )
    if not (formal / "run_conformance.py").is_file():
        missing.append("formal/run_conformance.py absent (EK5 conformance unconfigured)")
    return missing


def _run(cmd: Sequence[str], cwd: Path, timeout_s: int) -> Tuple[int, str]:
    try:
        proc = subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def run_formal_gates(repo_root: Path, report: Report, lake_timeout_s: int) -> Tuple[str, List[str]]:
    """Run the formal gates; returns (formal_tier, skipped_checks)."""
    formal = repo_root / "formal"
    skipped: List[str] = []
    ok = True
    lake_bin = resolve_lake_bin() or "lake"

    rc, out = _run([lake_bin, "build"], formal, lake_timeout_s)
    report.result("formal.lake_build", rc == 0, "lake build succeeded", f"lake build rc={rc}: {out.strip()[-300:]}")
    ok &= rc == 0

    lean_files = list_lean_files(formal)
    forbidden: List[str] = []
    theorems: List[str] = []
    modules: List[str] = []
    for lf in lean_files:
        txt = lf.read_text(encoding="utf-8", errors="replace")
        rel = lf.relative_to(formal)
        for lineno, tok in scan_lean_forbidden(txt):
            forbidden.append(f"{rel.as_posix()}:{lineno}:{tok}")
        theorems.extend(collect_lean_theorems(txt))
        modules.append(".".join(rel.with_suffix("").parts))
    report.result(
        "formal.no_sorry_native_decide",
        not forbidden and bool(lean_files),
        f"0 sorry/native_decide across {len(lean_files)} .lean files",
        f"{len(forbidden)} hits: {forbidden[:5]}" if lean_files else "no .lean files found under formal/",
    )
    ok &= not forbidden and bool(lean_files)

    with tempfile.TemporaryDirectory() as tmp:
        audit = Path(tmp) / "AxiomAudit.lean"
        lines = [f"import {m}" for m in sorted(modules)] + [f"#print axioms {t}" for t in theorems]
        audit.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        rc, out = _run([lake_bin, "env", "lean", str(audit)], formal, lake_timeout_s)
        verdict = evaluate_axiom_report(out, expected=theorems)
        axiom_ok = rc == 0 and verdict["ok"] and bool(theorems)
        report.result(
            "formal.print_axioms_allowlist",
            axiom_ok,
            f"{verdict['declarations']} theorems depend only on {sorted(ALLOWED_AXIOMS)}",
            f"rc={rc} violations={verdict['violations'][:5]} errors={verdict['errors'][:3]}",
        )
        ok &= axiom_ok

        control = Path(tmp) / "NegativeControl.lean"
        control.write_text(
            "axiom bogusAxiom : False\n"
            "theorem negCtlCustom : False := bogusAxiom\n"
            "theorem negCtlReduceBool : (1 : Nat) = 1 := by native_decide\n"
            "theorem negCtlSorry : (2 : Nat) = 3 := sorry\n"
            "#print axioms negCtlCustom\n#print axioms negCtlReduceBool\n#print axioms negCtlSorry\n",
            encoding="utf-8",
            newline="\n",
        )
        rc_c, out_c = _run([lake_bin, "env", "lean", str(control)], formal, lake_timeout_s)
        neg = evaluate_axiom_report(out_c)
        rejected = len(neg["violations"]) == 3
        report.result(
            "formal.negative_control",
            rejected,
            "custom axiom, Lean.ofReduceBool and sorryAx were all rejected by the allowlist",
            f"negative control NOT rejected (rc={rc_c}, violations={neg['violations']})",
        )
        ok &= rejected

    manifest = load_nanoda_manifest(formal)
    conformance = formal / "run_conformance.py"
    declarations = [str(d) for d in (manifest or {}).get("export_declarations", [])]
    nanoda_args = [str(a) for a in (manifest or {}).get("nanoda_args", [])]
    nanoda_bin = resolve_nanoda_bin(str((manifest or {}).get("nanoda_bin") or "nanoda_bin"))
    if manifest is None or nanoda_bin is None or not conformance.is_file() or not declarations or not nanoda_args:
        # Fail closed: an unconfigured external kernel or conformance gate is never a PASS or PASS_PARTIAL.
        report.add(
            "formal.nanoda_external_kernel",
            "NOT_RUN",
            "nanoda_bin / formal/run_conformance.py / formal/nanoda_manifest.json unresolved or unconfigured",
        )
        return FORMAL_TIER_NOT_RUN, ["nanoda_external_kernel", "ek5_1000_mutation_conformance"]

    prepared = True
    if export_is_stale(formal, manifest):
        if resolve_lean4export_bin(str(manifest.get("lean4export_bin") or "lean4export")) is None:
            report.add("formal.lean4export_prepare", "NOT_RUN", "export missing/stale and lean4export not found")
            return FORMAL_TIER_NOT_RUN, ["nanoda_external_kernel"]
        rc, out = _run([sys.executable, str(conformance), "--prepare-nanoda"], repo_root, lake_timeout_s)
        prepared = rc == 0 and not export_is_stale(formal, manifest)
        report.result(
            "formal.lean4export_prepare",
            prepared,
            "regenerated the lean4export file (it was missing or stale)",
            f"run_conformance.py --prepare-nanoda rc={rc}: {out.strip()[-300:]}",
        )
        ok &= prepared
    else:
        report.add("formal.lean4export_prepare", "PASS", "export is newer than every formal/**/*.lean and the manifest")

    if prepared:
        rc, out = _run([nanoda_bin, *nanoda_args], formal, lake_timeout_s)
        verdict = parse_nanoda_output(rc, out, len(declarations))
        report.result(
            "formal.nanoda_external_kernel",
            bool(verdict["accepted"]),
            f"external kernel checked {verdict['declarations_checked']} declarations "
            f"(>= {len(declarations)} exported) with no typechecker errors",
            f"nanoda rc={rc} checked={verdict['declarations_checked']} expected>={len(declarations)}: {out.strip()[-300:]}",
        )
        ok &= bool(verdict["accepted"])
    else:
        report.failed("formal.nanoda_external_kernel", "not run: the lean4export file could not be prepared")

    with tempfile.TemporaryDirectory(prefix="chelated_conformance_verify_") as _tmp_conf:
        tmp_conf_report = Path(_tmp_conf) / "conformance_report.json"
        rc, out = _run(
            [sys.executable, str(conformance), "--report", str(tmp_conf_report)],
            repo_root,
            lake_timeout_s,
        )
    report.result(
        "formal.ek5_1000_mutation_conformance",
        rc == 0,
        "1,000-sequence EK5 conformance passed",
        f"run_conformance.py rc={rc}: {out.strip()[-300:]}",
    )
    ok &= rc == 0

    if not ok:
        return FORMAL_TIER_FAIL, skipped
    return FORMAL_TIER_PASS, skipped


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_fast_checks(repo_root: Path, report: Report, d8_replay: bool) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    spec_path = repo_root / SPEC_REL
    if not spec_path.is_file():
        report.failed("registry.spec", f"{SPEC_REL} missing")
        return summary
    try:
        spec = load_json(spec_path)
    except VerifyError as exc:
        report.failed("registry.spec", str(exc))
        return summary
    verify_pristine_mirror(repo_root, spec, report)
    stats = verify_spec_bundles(repo_root, spec, report)
    summary["bundles_total"] = stats["total"]
    summary["bundles_verified"] = stats["verified"]
    summary["bundles_pending_optional"] = stats["pending"]
    summary["bundles_failed"] = stats["failed"]
    summary["part7_standalone_verified"] = verify_part7(repo_root, stats["dirs"], report)
    summary["d8"] = verify_d8(repo_root, report, d8_replay)
    summary["d11"] = verify_d11(repo_root, report)
    verify_spec_lock(repo_root, report)
    verify_non_nda(repo_root, report)
    verify_headline_metrics(repo_root, report)
    verify_registry_file(repo_root, spec, report)
    verify_release_status(repo_root, spec, report)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline verification of the whole evidence repository.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="CPU-only fast tier (default)")
    mode.add_argument("--full-formal", action="store_true", help="fast tier plus the Lean/Lake formal gates")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT, help="repository root to verify")
    parser.add_argument("--json-out", type=Path, default=None, help="write the machine-readable report here")
    parser.add_argument(
        "--no-d8-replay",
        action="store_true",
        help="skip the live D8 engine replay (reduced assurance; recorded as SKIPPED_BY_FLAG)",
    )
    parser.add_argument("--lake-timeout", type=int, default=7200, help="seconds allowed per formal subprocess")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    full = bool(args.full_formal)
    started = datetime.now(timezone.utc).isoformat()
    t0, cpu0 = time.perf_counter(), time.process_time()

    report = Report()
    summary = run_fast_checks(repo_root, report, d8_replay=not args.no_d8_replay)
    fast_failed = bool(report.failures)

    formal_tier = FORMAL_TIER_SKIPPED
    skipped = list(FAST_SKIPPED_CHECKS)
    if full:
        missing = formal_prerequisites(repo_root)
        if missing:
            formal_tier = FORMAL_TIER_NOT_RUN
            report.add("formal.prerequisites", "NOT_RUN", "; ".join(missing))
        else:
            formal_tier, skipped = run_formal_gates(repo_root, report, args.lake_timeout)

    failures = report.failures
    if failures:
        overall, rc = "FAIL", 1
    elif full and formal_tier == FORMAL_TIER_NOT_RUN:
        overall, rc = "NOT_RUN", 2
    else:
        overall, rc = "PASS", 0

    wall = time.perf_counter() - t0
    payload = {
        "tool": "scripts/verify_all_offline.py",
        "mode": "full-formal" if full else "fast",
        "started_utc": started,
        "repo_root": str(repo_root),
        "wall_clock_seconds": round(wall, 3),
        "cpu_seconds_this_process": round(time.process_time() - cpu0, 3),
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "python": platform.python_version(),
        "formal_tier": formal_tier,
        "skipped_checks": skipped,
        "fast_tier_failed": fast_failed,
        "overall": overall,
        "exit_code": rc,
        "summary": summary,
        "checks": report.checks,
    }

    for row in report.checks:
        print(f"[{row['status']:<7}] {row['name']}: {row['detail']}")
    print(
        f"verify_all_offline: {overall} mode={payload['mode']} formal_tier={formal_tier} "
        f"wall={wall:.1f}s cpu={payload['cpu_seconds_this_process']}s "
        f"platform={payload['platform']['system']} python={payload['python']} "
        f"skipped_checks={','.join(skipped) if skipped else 'none'}"
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
