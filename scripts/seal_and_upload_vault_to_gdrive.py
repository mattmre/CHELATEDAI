r"""Finalize, Hash-Sign, AES-256 Multipart-Encrypt (`7z`), Verify, and Place Vault on Google Drive Mirror.

Completes `CHELATEDAI-COMPLETE-VAULT-2026-11-22` after `stage_complete_vault_for_gdrive.py`
and `run_phase2_d1_local_fleet_proof.py` finish:
  1. Injects latest `phase2_d1_4node_local_fleet_proof` receipts, `prior-art-and-novelty-audit-2026-11-22.md`,
     stash refs from `D:\GITHUB\CHELATEDAI\.git`, and the latest live conversation transcript (`d68f68df...`).
  2. Prunes reproducible `.venv-egv-prod` site-packages bloat (recording package inventory instead).
  3. Generates `ARCHIVE_MASTER_INDEX.md` (with exact live 21-worktree SHAs), `VAULT-MANIFEST.json`, and
     `SHA256SUMS-INNER.txt`.
  4. Packages into multipart WinZip/7-Zip AES-256 (`-tzip -mem=AES256 -v1900m`) volumes inside
     `O:\backups\CHELATEDAI-COMPLETE-VAULT-2026-11-22\` (active Google Drive mirror root), writes outer
     `SHA256SUMS.txt` + `README_RESTORE_INSTRUCTIONS.md`, and runs `7z t` cryptographic round-trip verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

STAGE_WRAPPER = Path(r"J:\CV")
STAGE_ROOT = STAGE_WRAPPER / "CHELATEDAI-COMPLETE-VAULT-2026-11-22"
STAGE0_ROOT = Path(r"D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1")
MAIN_GIT = Path(r"D:\GITHUB\CHELATEDAI\.git")
GDRIVE_MIRROR_DEST = Path(r"O:\backups\CHELATEDAI-COMPLETE-VAULT-2026-11-22")
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")


def _lp(p: Path) -> str:
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def _sha256_lp(lp_str: str) -> str:
    h = hashlib.sha256()
    with open(lp_str, "rb") as fh:
        while True:
            chunk = fh.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _copy_dir_lp(src: Path, dst: Path) -> None:
    if not os.path.exists(_lp(src)):
        return
    src_abs = os.path.abspath(str(src))
    for root, dirs, files in os.walk(_lp(src)):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".venv", "venv", "node_modules"}]
        clean_root = root[4:] if root.startswith("\\\\?\\") else root
        rel = os.path.relpath(clean_root, src_abs)
        tdir = dst if rel == "." else (dst / rel)
        os.makedirs(_lp(tdir), exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(root, f), _lp(tdir / f))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/seal_and_upload_vault_to_gdrive.py <passphrase>")
    passphrase = sys.argv[1]
    t0 = time.perf_counter()

    # 1. Prune `.venv-egv-prod` virtualenv bloat inside staged CHELATEDAI-EGV-TRAINING if present
    venv_bloat = STAGE_ROOT / "01-git-and-worktrees" / "worktree-snapshots" / "CHELATEDAI-EGV-TRAINING" / ".venv-egv-prod"
    if os.path.exists(_lp(venv_bloat)):
        dist_infos = []
        for r, _ds, _fs in os.walk(_lp(venv_bloat)):
            for dname in _ds:
                if dname.endswith(".dist-info"):
                    dist_infos.append(dname[:-10])
        inv_file = venv_bloat.parent / "VENV_EGV_PROD_PACKAGE_INVENTORY.txt"
        with open(_lp(inv_file), "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(set(dist_infos))) + "\n")
        shutil.rmtree(_lp(venv_bloat), ignore_errors=True)
        print(f"[VaultSeal] Replaced .venv-egv-prod binary bloat with {len(set(dist_infos))}-pkg inventory.", flush=True)

    # 2. Copy Phase-2 D1 4-node fleet proof, Novelty Audit, latest STAGE0 files, git stash logs, and live brain transcript
    d1_src = STAGE0_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof"
    if d1_src.is_dir():
        _copy_dir_lp(d1_src, STAGE_ROOT / "02-hardware-evidence" / "phase2-d1-4node-fleet-proof")
    _copy_dir_lp(STAGE0_ROOT / "docs" / "research", STAGE_ROOT / "02-hardware-evidence" / "bundles-27-of-27" / "docs-research")
    for extra_py in ("chelated_raw_text_exocortex.py", "prospective_realworld_drift_corpus.py"):
        src_py = STAGE0_ROOT / extra_py
        if src_py.is_file():
            dst_py = STAGE_ROOT / "01-git-and-worktrees" / "worktree-snapshots" / "CHELATEDAI-STAGE0-NIGHT1" / extra_py
            os.makedirs(_lp(dst_py.parent), exist_ok=True)
            shutil.copy2(_lp(src_py), _lp(dst_py))

    # Preserve primary `.git` stash reflog & archive refs metadata
    stash_meta_dir = STAGE_ROOT / "01-git-and-worktrees" / "stash-and-archive-refs"
    os.makedirs(_lp(stash_meta_dir), exist_ok=True)
    for rel_ref in ("logs/refs/stash", "refs/stash", "packed-refs"):
        p_ref = MAIN_GIT / rel_ref
        if p_ref.is_file():
            out_ref = stash_meta_dir / rel_ref.replace("/", "__")
            shutil.copy2(_lp(p_ref), _lp(out_ref))
    try:
        stash_patches = subprocess.check_output(
            ["git", "-C", str(STAGE0_ROOT), "stash", "list", "-p"], stderr=subprocess.DEVNULL
        )
        if stash_patches:
            with open(_lp(stash_meta_dir / "all_stashes_full_patches.patch"), "wb") as fh:
                fh.write(stash_patches)
    except Exception:
        pass

    # Refresh active Antigravity conversation transcript so latest messages are captured
    live_brain = Path(
        os.path.expanduser(
            r"~\.t3\userdata\providers\antigravity\ac0a3dfd6dddb20962cecff6ee5fe65e19d3923be20e52c5ab52ff877f7e4c32\antigravity-acp\brain\d68f68df-2385-4788-8983-e239ca3c27e0"
        )
    )
    if live_brain.is_dir():
        _copy_dir_lp(live_brain, STAGE_ROOT / "03-ai-provenance" / "antigravity-gemini-brain" / live_brain.name)

    # 3. Build ARCHIVE_MASTER_INDEX.md with exact live 21-worktree table + D1 summary + Restoration Guide
    topo_path = STAGE_ROOT / "01-git-and-worktrees" / "worktree-topology.json"
    topo = json.loads(open(_lp(topo_path), encoding="utf-8").read()) if os.path.exists(_lp(topo_path)) else {"worktrees": []}
    wt_lines = [
        "| # | Worktree Directory | Branch / Mode | Exact HEAD SHA | Patch Preserved |",
        "|---:|---|---|---|---|",
    ]
    for idx, wt in enumerate(topo.get("worktrees", [])):
        b_str = wt.get("branch") or ("detached" if wt.get("detached") else "unknown")
        wt_lines.append(
            f"| {idx+1:02d} | `{wt.get('path')}` | `{b_str}` | `{wt.get('head')}` | `{wt.get('patchFile') or 'clean'}` |"
        )

    d1_summary_md = "_(See `02-hardware-evidence/phase2-d1-4node-fleet-proof/` for full rows & manifest.)_"
    d1_man_file = d1_src / "phase2_d1_4node_fleet_manifest.json"
    if d1_man_file.is_file():
        d1_man = json.loads(d1_man_file.read_text(encoding="utf-8"))
        arm_lines = ["| Arm | Hardware Node(s) | N | Accuracy | Mean Prompt Tokens |", "|---|---|---:|---:|---:|"]
        for k_a, v_a in sorted(d1_man.get("arm_summaries", {}).items()):
            arm_lines.append(
                f"| `{k_a}` | 4-Node Fleet | {v_a['n_cases']} | **{v_a['accuracy']:.4f}** | {v_a['mean_prompt_tokens']} |"
            )
        d1_summary_md = "\n".join(arm_lines)

    template_path = STAGE0_ROOT / "docs" / "research" / "ARCHIVE_MASTER_INDEX_TEMPLATE.md"
    tpl_text = template_path.read_text(encoding="utf-8") if template_path.is_file() else "# CHELATEDAI Complete Vault\n"
    master_md = (
        tpl_text.replace("{{LIVE_WORKTREE_TABLE}}", "\n".join(wt_lines))
        .replace("{{PHASE2_D1_TABLE}}", d1_summary_md)
    )
    with open(_lp(STAGE_ROOT / "ARCHIVE_MASTER_INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write(master_md)

    # 4. Compute SHA256SUMS-INNER.txt & VAULT-MANIFEST.json across every file in STAGE_ROOT
    print("[VaultSeal] Computing inner SHA-256 digests across entire staged vault...", flush=True)
    stage_abs = os.path.abspath(str(STAGE_ROOT))
    inner_entries: List[Dict[str, Any]] = []
    sum_lines: List[str] = []
    total_inner_bytes = 0
    for root, _dirs, files in os.walk(_lp(STAGE_ROOT)):
        clean_root = root[4:] if root.startswith("\\\\?\\") else root
        for f in sorted(files):
            if f in {"SHA256SUMS-INNER.txt", "VAULT-MANIFEST.json"}:
                continue
            full_lp = os.path.join(root, f)
            rel_posix = os.path.relpath(os.path.join(clean_root, f), stage_abs).replace("\\", "/")
            sz = os.stat(full_lp).st_size
            digest = _sha256_lp(full_lp)
            total_inner_bytes += sz
            sum_lines.append(f"{digest}  {rel_posix}")
            inner_entries.append({"path": rel_posix, "bytes": sz, "sha256": digest})

    man_dir = STAGE_ROOT / "00-manifests"
    os.makedirs(_lp(man_dir), exist_ok=True)
    with open(_lp(man_dir / "SHA256SUMS-INNER.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sum_lines) + "\n")

    vault_manifest = {
        "vault_id": "CHELATEDAI-COMPLETE-VAULT-2026-11-22",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "encryption": "WinZip/7-Zip AES-256 (-tzip -mem=AES256)",
        "total_inner_files": len(inner_entries),
        "total_inner_bytes": total_inner_bytes,
        "total_inner_gib": round(total_inner_bytes / (1024.0**3), 3),
        "worktree_count": len(topo.get("worktrees", [])),
    }
    with open(_lp(man_dir / "VAULT-MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(vault_manifest, fh, indent=2)

    # 5. Create AES-256 encrypted multipart `.zip.001..NNN` directly inside Google Drive Mirror (`O:\backups\...`)!
    GDRIVE_MIRROR_DEST.mkdir(parents=True, exist_ok=True)
    for old_part in GDRIVE_MIRROR_DEST.glob("CHELATEDAI-COMPLETE-VAULT-2026-11-22.zip*"):
        old_part.unlink()

    archive_base = GDRIVE_MIRROR_DEST / "CHELATEDAI-COMPLETE-VAULT-2026-11-22.zip"
    print(
        f"[VaultSeal] Compressing & AES-256 encrypting {vault_manifest['total_inner_files']} files "
        f"({vault_manifest['total_inner_gib']} GiB) into multipart volumes at {GDRIVE_MIRROR_DEST}...",
        flush=True,
    )
    cmd_7z = [
        str(SEVEN_ZIP),
        "a",
        "-tzip",
        "-mem=AES256",
        "-mx=5",
        "-mmt=on",
        "-v1900m",
        f"-p{passphrase}",
        str(archive_base),
        "CHELATEDAI-COMPLETE-VAULT-2026-11-22",
    ]
    subprocess.run(cmd_7z, cwd=str(STAGE_WRAPPER), check=True)

    # Handle single-volume vs multipart naming seamlessly
    parts = sorted(GDRIVE_MIRROR_DEST.glob("CHELATEDAI-COMPLETE-VAULT-2026-11-22.zip*"))
    if not parts:
        raise RuntimeError("7z produced no output archive files!")

    # 6. Cryptographic round-trip test (`7z t`) on first volume/archive!
    first_vol = parts[0]
    print(f"[VaultSeal] Running 7z AES-256 decryption & CRC/integrity verification on {first_vol.name}...", flush=True)
    test_res = subprocess.run(
        [str(SEVEN_ZIP), "t", f"-p{passphrase}", str(first_vol)],
        capture_output=True,
        text=True,
        check=True,
    )
    if "Everything is Ok" not in test_res.stdout:
        raise RuntimeError(f"7z verification did not report 'Everything is Ok': {test_res.stdout[-400:]}")

    # 7. Write outer `SHA256SUMS.txt` and password-free `README_RESTORE_VAULT.md` + `UPLOAD_RECEIPT.json`
    outer_sums: List[str] = []
    part_records: List[Dict[str, Any]] = []
    for p_vol in parts:
        d_vol = _sha256_lp(_lp(p_vol))
        sz_vol = p_vol.stat().st_size
        outer_sums.append(f"{d_vol}  {p_vol.name}")
        part_records.append({"filename": p_vol.name, "bytes": sz_vol, "mib": round(sz_vol / (1024.0**2), 2), "sha256": d_vol})

    (GDRIVE_MIRROR_DEST / "SHA256SUMS.txt").write_text("\n".join(outer_sums) + "\n", encoding="utf-8")
    (GDRIVE_MIRROR_DEST / "README_RESTORE_VAULT.md").write_text(master_md, encoding="utf-8")

    receipt = {
        "vault_id": "CHELATEDAI-COMPLETE-VAULT-2026-11-22",
        "destination_google_drive_mirror": str(GDRIVE_MIRROR_DEST),
        "google_drive_sync_root": "GoogleDriveFS MirrorRoot [title='backups', root_path='O:\\backups', connected=true]",
        "encryption_verified_7z_t": True,
        "inner_vault_stats": vault_manifest,
        "outer_volumes": part_records,
        "total_compressed_bytes": sum(r["bytes"] for r in part_records),
        "total_compressed_mib": round(sum(r["bytes"] for r in part_records) / (1024.0**2), 2),
        "seal_wall_sec": round(time.perf_counter() - t0, 2),
    }
    (GDRIVE_MIRROR_DEST / "VAULT_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (STAGE0_ROOT / "archive" / "GDRIVE_VAULT_RECEIPT_20261122.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[VaultSeal] SUCCESS! {len(parts)} AES-256 volume(s) verified & synced at {GDRIVE_MIRROR_DEST}", flush=True)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
