"""Single-source cryptographic verification registry & CI gate for Doctoral Whitepaper v2 & HTML v2.

Implements the R3 Council pipeline:
  discover_attempts -> verify_bytes -> normalize_observations -> validate_claims (--check)

Usage:
  python scripts/build_whitepaper_assets.py          # Build/update docs/research/verified_registry.json
  python scripts/build_whitepaper_assets.py --check  # Fail-closed CI gate over registry + Whitepaper v2 + HTML v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "docs/research/bundle_registry_spec.json"
REGISTRY_PATH = REPO_ROOT / "docs/research/verified_registry.json"
WHITEPAPER_PATH = REPO_ROOT / "docs/research/doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md"
HTML_PATH = REPO_ROOT / "docs/research/chelated-memory-spherical-migration-whitepaper-presentation.html"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pristine_mirror(manifest_rel: str) -> Dict[str, Any]:
    mpath = REPO_ROOT / manifest_rel
    if not mpath.exists():
        raise FileNotFoundError(f"Pristine mirror manifest missing: {mpath}")
    data = json.loads(mpath.read_text(encoding="utf-8"))
    mirror_dir = mpath.parent
    files = data.get("files", {})
    verified = 0
    mismatches: List[str] = []
    # Support both dict-of-path->meta and list-of-record schemas
    if isinstance(files, dict):
        items = [(k, v["sha256"] if isinstance(v, dict) else str(v)) for k, v in files.items()]
    else:
        items = [(str(entry["path"]), str(entry["sha256"])) for entry in files]
    for rel, expected_sha in items:
        fpath = mirror_dir / rel
        if not fpath.exists():
            mismatches.append(f"MISSING:{rel}")
            continue
        actual = sha256_file(fpath)
        if actual != expected_sha:
            mismatches.append(f"MISMATCH:{rel}:{actual[:12]}!={expected_sha[:12]}")
        else:
            verified += 1
    if mismatches:
        raise RuntimeError(f"Pristine mirror integrity failure: {mismatches}")
    return {
        "manifest_path": manifest_rel,
        "git_tag": data.get("git_tag", "baseline/pre-fix-pristine-2026-11-21"),
        "commit_sha": data.get("commit_sha", "b37e6a5523f8647c0d682178628db1f1545dc1ed"),
        "files_verified": verified,
        "files_total": len(items),
        "status": "VERIFIED_BYTE_EXACT",
    }


def verify_bundle(entry: Dict[str, Any]) -> Dict[str, Any]:
    bdir = REPO_ROOT / entry["dir"]
    mpath = bdir / entry["manifest"]
    rpath = bdir / entry["receipt"]
    if not rpath.exists() or not mpath.exists():
        if entry.get("optional_until_run", False):
            return {
                "id": entry["id"],
                "part": entry["part"],
                "role": entry["role"],
                "dir": entry["dir"],
                "present": False,
                "status": "PENDING_RUN",
                "notes": entry.get("notes", ""),
            }
        raise FileNotFoundError(f"Required bundle missing: {entry['id']} at {bdir}")

    receipt = json.loads(rpath.read_text(encoding="utf-8"))
    actual_manifest_sha = sha256_file(mpath)
    expected_manifest_sha = str(receipt["manifest_sha256"])
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            f"Manifest SHA-256 mismatch for {entry['id']}: actual={actual_manifest_sha} != receipt={expected_manifest_sha}"
        )

    # Verify ancillary hashes (S1 style)
    verified_side_hashes: Dict[str, str] = {}
    anc_map = {
        "m2_preflight_sha256": "s1_m2_preflight.json",
        "selection_lock_sha256": "selection_lock.json",
    }
    for rkey, fname in anc_map.items():
        if rkey in receipt:
            fpath = bdir / fname
            actual = sha256_file(fpath)
            if actual != str(receipt[rkey]):
                raise RuntimeError(f"Ancillary hash mismatch in {entry['id']}/{fname}")
            verified_side_hashes[fname] = actual

    # Verify R3 (`artifact_sha256`) or R4 (`artifact_hashes`) side-artifact hashes
    for dict_key in ("artifact_sha256", "artifact_hashes"):
        if dict_key in receipt and isinstance(receipt[dict_key], dict):
            for fname, exp_sha in receipt[dict_key].items():
                fpath = bdir / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"Receipt-listed artifact missing: {fpath}")
                actual = sha256_file(fpath)
                if actual != str(exp_sha):
                    raise RuntimeError(f"Side-artifact SHA-256 mismatch in {entry['id']}/{fname}")
                if fname != entry["manifest"]:
                    verified_side_hashes[fname] = actual

    # Transparently inventory any unhashed raw CSV/JSONL files (NEVER forge historical receipts!)
    unhashed_raw_files: Dict[str, Dict[str, Any]] = {}
    for fpath in sorted(bdir.iterdir()):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() in (".csv", ".jsonl"):
            if fpath.name not in verified_side_hashes:
                unhashed_raw_files[fpath.name] = {
                    "status": "UNHASHED_IN_HISTORICAL_RECEIPT",
                    "publication_pinned_sha256": sha256_file(fpath),
                    "bytes": fpath.stat().st_size,
                }

    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    return {
        "id": entry["id"],
        "part": entry["part"],
        "role": entry["role"],
        "dir": entry["dir"],
        "manifest_file": entry["manifest"],
        "present": True,
        "status": "SHA256_VERIFIED",
        "manifest_sha256": actual_manifest_sha,
        "git_sha": manifest.get("git_sha", ""),
        "wall_seconds": manifest.get("wall_seconds", manifest.get("runtime_seconds")),
        "verified_side_artifacts": verified_side_hashes,
        "unhashed_historical_raw_files": unhashed_raw_files,
        "notes": entry.get("notes", ""),
        "_manifest_obj": manifest,
    }


def extract_key_metrics(bundles_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize canonical observations from verified manifests into a clean summary ledger."""
    obs: Dict[str, Any] = {}

    if "s1" in bundles_by_id and bundles_by_id["s1"]["present"]:
        m = bundles_by_id["s1"]["_manifest_obj"]
        obs["s1"] = {"wall_seconds": m.get("wall_seconds"), "verdict": m.get("overall_verdict", m.get("verdict"))}

    if "s3" in bundles_by_id and bundles_by_id["s3"]["present"]:
        m = bundles_by_id["s3"]["_manifest_obj"]
        obs["s3"] = {
            "verdict": m.get("overall_verdict", m.get("verdict")),
            "wrong_space_blocks": m.get("wrong_space_audit", {}).get("blocked_count"),
        }

    if "f2_longmemeval_v2" in bundles_by_id and bundles_by_id["f2_longmemeval_v2"]["present"]:
        m = bundles_by_id["f2_longmemeval_v2"]["_manifest_obj"]
        obs["f2_longmemeval_v2"] = {
            "wall_seconds": m.get("wall_seconds"),
            "regimes": m.get("regimes", m.get("summary")),
        }

    if "f2_ek5_v2_1m" in bundles_by_id and bundles_by_id["f2_ek5_v2_1m"]["present"]:
        m = bundles_by_id["f2_ek5_v2_1m"]["_manifest_obj"]
        obs["f2_ek5_v2_1m"] = {"wall_seconds": m.get("wall_seconds")}

    if "f2_unified_zs_locomo" in bundles_by_id and bundles_by_id["f2_unified_zs_locomo"]["present"]:
        m = bundles_by_id["f2_unified_zs_locomo"]["_manifest_obj"]
        obs["f2_unified_zs_locomo"] = {
            "wall_seconds": m.get("wall_seconds"),
            "zero_shortcut": m.get("part_a_zero_shortcut", {}).get("arms"),
        }

    if "f1_4096_ood" in bundles_by_id and bundles_by_id["f1_4096_ood"]["present"]:
        m = bundles_by_id["f1_4096_ood"]["_manifest_obj"]
        obs["f1_4096_ood"] = {"wall_seconds": m.get("wall_seconds")}

    if "f1_r4_comprehensive" in bundles_by_id and bundles_by_id["f1_r4_comprehensive"]["present"]:
        m = bundles_by_id["f1_r4_comprehensive"]["_manifest_obj"]
        obs["f1_r4_comprehensive"] = {
            "wall_seconds": m.get("wall_seconds"),
            "n_per_query_rows": m.get("n_per_query_rows"),
            "n_summary_rows": m.get("n_summary_rows"),
            "suite_1a_4096_ood": m.get("suite_1a_4096_ood"),
            "suite_1b_beir_and_b4": m.get("suite_1b_beir_and_b4"),
        }

    if "f2_ek5_v3_1m_powerlaw" in bundles_by_id and bundles_by_id["f2_ek5_v3_1m_powerlaw"]["present"]:
        m = bundles_by_id["f2_ek5_v3_1m_powerlaw"]["_manifest_obj"]
        obs["f2_ek5_v3_1m_powerlaw"] = {
            "wall_seconds": m.get("wall_seconds"),
            "differential_oracle_proofs": m.get("differential_oracle_proofs"),
            "scale_1m_kernel": m.get("scale_1m_kernel"),
            "service_100k_end_to_end": m.get("service_100k_end_to_end"),
        }

    if "f2_r4_zero_shortcut_v2_and_locomo" in bundles_by_id and bundles_by_id["f2_r4_zero_shortcut_v2_and_locomo"]["present"]:
        m = bundles_by_id["f2_r4_zero_shortcut_v2_and_locomo"]["_manifest_obj"]
        obs["f2_r4_zero_shortcut_v2_and_locomo"] = {
            "wall_seconds": m.get("wall_seconds"),
            "part_a_zero_shortcut_v2": m.get("part_a_zero_shortcut_v2"),
            "part_b_locomo_300_balanced": m.get("part_b_locomo_300_balanced"),
        }

    return obs


def build_registry() -> Dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    pristine = verify_pristine_mirror(spec["pristine_mirror_manifest"])

    verified_bundles: List[Dict[str, Any]] = []
    bundles_by_id: Dict[str, Dict[str, Any]] = {}
    total_side_verified = 0
    total_unhashed_hist = 0
    for entry in spec["bundles"]:
        vb = verify_bundle(entry)
        bundles_by_id[vb["id"]] = vb
        if vb["present"]:
            total_side_verified += len(vb["verified_side_artifacts"])
            total_unhashed_hist += len(vb["unhashed_historical_raw_files"])
        clean_vb = {k: v for k, v in vb.items() if k != "_manifest_obj"}
        verified_bundles.append(clean_vb)

    observations = extract_key_metrics(bundles_by_id)
    present_count = sum(1 for b in verified_bundles if b["present"])
    payload: Dict[str, Any] = {
        "schema_version": "verified_registry_v2",
        "release_status": spec["release_status"],
        "pristine_mirror": pristine,
        "counts": {
            "bundles_total_spec": len(verified_bundles),
            "bundles_present_verified": present_count,
            "verified_side_artifacts": total_side_verified,
            "unhashed_historical_raw_files": total_unhashed_hist,
        },
        "bundles": verified_bundles,
        "observations": observations,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["registry_content_sha256"] = digest
    return payload


def validate_publication_documents(registry: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not WHITEPAPER_PATH.exists():
        errors.append(f"Missing Whitepaper v2: {WHITEPAPER_PATH}")
    if not HTML_PATH.exists():
        errors.append(f"Missing Presentation HTML v2: {HTML_PATH}")
    if errors:
        return False, errors

    wp = WHITEPAPER_PATH.read_text(encoding="utf-8")
    html = HTML_PATH.read_text(encoding="utf-8")

    reg_sha12 = registry["registry_content_sha256"][:12]
    for label, doc in (("Whitepaper", wp), ("HTML", html)):
        if "PRE-RELEASE" not in doc:
            errors.append(f"{label} missing mandatory PRE-RELEASE status banner")
        if re.search(r"\bShip-Ready\b", doc, re.IGNORECASE):
            errors.append(f"{label} contains forbidden promotional phrase 'Ship-Ready'")
        if reg_sha12 not in doc:
            errors.append(f"{label} missing current verified_registry content SHA prefix {reg_sha12}")
        if "TRA1_OR_M4_NEEDS_REVIEW" not in doc or "-0.045658" not in doc:
            errors.append(f"{label} missing mandatory S3 honest disclosure (TRA1_OR_M4_NEEDS_REVIEW & -0.045658)")
        if "42/50" not in doc or "46/50" not in doc:
            errors.append(f"{label} missing mandatory LongMemEval Regime A clean 42/50 vs 46/50 disclosure")
        if "117/3" not in doc or "3-key" not in doc:
            errors.append(f"{label} missing mandatory R3 LoCoMo pre-fix audit baseline disclosure")
        # Check every present bundle manifest SHA prefix appears in Whitepaper ledger
        if label == "Whitepaper":
            for b in registry["bundles"]:
                if b["present"] and b["manifest_sha256"][:12] not in doc:
                    errors.append(f"Whitepaper missing verified SHA-256 prefix for bundle {b['id']} ({b['manifest_sha256'][:12]})")

    # HTML-specific R3 accessibility & print-integrity assertions
    if "--warm: #7d2f0a" not in html:
        errors.append("HTML missing R3-locked WCAG AAA --warm: #7d2f0a")
    if "#c95d28" in html.lower():
        errors.append("HTML still contains failed-contrast #c95d28 token")
    if "<math" not in html:
        errors.append("HTML missing native MathML (<math>) equations")
    if 'role="tablist"' not in html:
        errors.append("HTML missing ARIA role='tablist' interactive tabs")
    if ".table-scroll" not in html:
        errors.append("HTML missing responsive 375px .table-scroll wrapper")
    if "release-status-banner" not in html:
        errors.append("HTML missing print-visible .release-status-banner inside hero")
    if html.count("<svg") < 6:
        errors.append(f"HTML expected >= 6 static inline SVG figures, found {html.count('<svg')}")

    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or check verified publication registry and documents.")
    parser.add_argument("--check", action="store_true", help="Fail-closed CI verification mode")
    args = parser.parse_args()

    reg = build_registry()
    rendered = json.dumps(reg, indent=2, sort_keys=True, ensure_ascii=True) + "\n"

    if args.check:
        if not REGISTRY_PATH.exists():
            print("FAIL: docs/research/verified_registry.json does not exist; run without --check first.", file=sys.stderr)
            sys.exit(1)
        if REGISTRY_PATH.read_text(encoding="utf-8") != rendered:
            print("FAIL: docs/research/verified_registry.json is stale relative to on-disk receipts.", file=sys.stderr)
            sys.exit(1)
        ok, errs = validate_publication_documents(reg)
        if not ok:
            for e in errs:
                print(f"FAIL: {e}", file=sys.stderr)
            sys.exit(1)
        print(
            f"WHITEPAPER_ASSETS_CHECK: PASS "
            f"(bundles={reg['counts']['bundles_present_verified']}/{reg['counts']['bundles_total_spec']}, "
            f"side_hashes={reg['counts']['verified_side_artifacts']}, "
            f"unhashed_hist_raw={reg['counts']['unhashed_historical_raw_files']}, "
            f"pristine={reg['pristine_mirror']['files_verified']}/{reg['pristine_mirror']['files_total']}, "
            f"registry_sha={reg['registry_content_sha256'][:12]})"
        )
        return

    REGISTRY_PATH.write_text(rendered, encoding="utf-8")
    print(
        f"WROTE {REGISTRY_PATH} "
        f"(bundles={reg['counts']['bundles_present_verified']}/{reg['counts']['bundles_total_spec']}, "
        f"side_hashes={reg['counts']['verified_side_artifacts']}, "
        f"unhashed_hist_raw={reg['counts']['unhashed_historical_raw_files']}, "
        f"pristine={reg['pristine_mirror']['files_verified']}/{reg['pristine_mirror']['files_total']}, "
        f"registry_sha={reg['registry_content_sha256'][:12]})"
    )


if __name__ == "__main__":
    main()
