r"""Stage Complete CHELATEDAI Research, 21-Worktree, Spark Telemetry, & Multi-AI Provenance Vault.

Read-only collection script that stages:
  1. Complete Git object bundle (`--all`) + declarative 21-worktree topology & uncommitted diffs
     + full source/docs/artifacts/archives across all `D:\GITHUB\CHELATEDAI*` directories
     (skipping only Python/Node virtualenvs `.venv`, `venv`, `node_modules`, `__pycache__`, `.pytest_cache`).
  2. Hardware benchmark receipts across `CHELATEDAI-STAGE0-NIGHT1` (`27/27` bundles + local `archive/` vaults)
     plus remote DGX Spark `out/` manifests & CSVs from `gx10-37c3` (`10.0.0.49`) and `gx10-6724` (`10.0.0.194`).
  3. Multi-Engine AI Provenance for `CHELATEDAI`:
     - T3 Code `statev2.sqlite` & `state.sqlite` safe online SQLite backup + `CHELATEDAI` thread JSON exports + `userdata\logs`
     - Antigravity / Gemini 4 Neon conversation brains matching `CHELATED` (including current session & subagents)
     - Anthropic Claude Code `.claude\projects\*CHELATED*` session logs
     - OpenAI Codex `.codex\sessions` & `archived_sessions` JSONL transcripts referencing `CHELATED` / `chelated-night1`
     - Frontier Council & 4-Model Panel verbatim transcripts (`Astra Ultra`, `Claude Opus Ultrathink`, `Grok XHigh`, `Neon XHigh`)
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

STAGE_ROOT = Path(r"J:\CV\CHELATEDAI-COMPLETE-VAULT-2026-11-22")
GITHUB_ROOT = Path(r"D:\GITHUB")
STAGE0_ROOT = Path(r"D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1")
USER_HOME = Path(os.path.expanduser("~"))

SKIP_DIR_NAMES: Set[str] = {
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "cef_cache",
}


def _lp(p: Path) -> str:
    """Return Windows extended-length `\\\\?\\` path string so 260-char MAX_PATH never trips."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def _copy_tree_filtered(src: Path, dst: Path) -> Tuple[int, int]:
    """Copy directory tree read-only, skipping transient virtualenv/cache dirs; return (files, bytes)."""
    n_files = 0
    n_bytes = 0
    if not os.path.exists(_lp(src)):
        return 0, 0
    src_abs = os.path.abspath(str(src))
    for root, dirs, files in os.walk(_lp(src)):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        clean_root = root[4:] if root.startswith("\\\\?\\") else root
        rel = os.path.relpath(clean_root, src_abs)
        target_dir = dst if rel == "." else (dst / rel)
        os.makedirs(_lp(target_dir), exist_ok=True)
        for fname in files:
            s_lp = os.path.join(root, fname)
            d_lp = _lp(target_dir / fname)
            try:
                s_st = os.stat(s_lp)
                if not os.path.exists(d_lp) or os.stat(d_lp).st_size != s_st.st_size:
                    shutil.copy2(s_lp, d_lp)
                n_files += 1
                n_bytes += s_st.st_size
            except Exception:
                pass
    return n_files, n_bytes


def _safe_sqlite_backup(src_db: Path, dst_db: Path) -> bool:
    if not src_db.is_file():
        return False
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    try:
        src_con = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True, timeout=15.0)
        dst_con = sqlite3.connect(str(dst_db))
        with dst_con:
            src_con.backup(dst_con)
        dst_con.close()
        src_con.close()
        return True
    except Exception:
        try:
            shutil.copy2(src_db, dst_db)
            return True
        except Exception:
            return False


def stage_git_and_worktrees() -> Dict[str, Any]:
    out_git = STAGE_ROOT / "01-git-and-worktrees"
    patches_dir = out_git / "worktree-patches"
    snapshots_dir = out_git / "worktree-snapshots"
    patches_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create full all-refs Git bundle from CHELATEDAI-STAGE0-NIGHT1 (shares object store with CHELATEDAI)
    bundle_path = out_git / "chelatedai-all-refs.bundle"
    subprocess.run(
        ["git", "-C", str(STAGE0_ROOT), "bundle", "create", str(bundle_path), "--all"],
        check=True,
    )
    verify_res = subprocess.run(
        ["git", "bundle", "verify", str(bundle_path)],
        capture_output=True,
        text=True,
        cwd=str(out_git),
    )

    # 2. Parse `git worktree list --porcelain`
    wt_raw = subprocess.check_output(
        ["git", "-C", str(STAGE0_ROOT), "worktree", "list", "--porcelain"],
        text=True,
    )
    worktrees: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    for line in wt_raw.splitlines():
        if not line.strip():
            if cur:
                worktrees.append(cur)
                cur = {}
            continue
        if line.startswith("worktree "):
            cur["path"] = line[len("worktree ") :].strip()
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD ") :].strip()
        elif line.startswith("branch "):
            b_ref = line[len("branch ") :].strip()
            cur["branch"] = b_ref.replace("refs/heads/", "")
            cur["detached"] = False
        elif line.strip() == "detached":
            cur["detached"] = True
            cur["branch"] = None
    if cur:
        worktrees.append(cur)

    for idx, wt in enumerate(worktrees):
        wt_path = Path(wt["path"])
        wt["name"] = wt_path.name if idx > 0 else "CHELATEDAI"
        wt["isPrimary"] = idx == 0
        wt["patchFile"] = None
        if wt_path.is_dir():
            try:
                diff_bytes = subprocess.check_output(["git", "-C", str(wt_path), "diff", "HEAD"], stderr=subprocess.DEVNULL)
                if diff_bytes.strip():
                    pname = f"{idx:02d}_{wt_path.name}.patch"
                    (patches_dir / pname).write_bytes(diff_bytes)
                    wt["patchFile"] = pname
            except Exception:
                pass

    (out_git / "worktree-topology.json").write_text(
        json.dumps({"worktree_count": len(worktrees), "worktrees": worktrees}, indent=2),
        encoding="utf-8",
    )

    # 3. Copy all D:\GITHUB\CHELATEDAI* directories (preserving untracked archives, artifacts, reports, configs!)
    total_files, total_bytes = 0, 0
    copied_dirs: List[str] = []
    for d in sorted(GITHUB_ROOT.glob("*CHELATED*")):
        if d.is_dir():
            nf, nb = _copy_tree_filtered(d, snapshots_dir / d.name)
            total_files += nf
            total_bytes += nb
            copied_dirs.append(d.name)

    return {
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_verify_rc": verify_res.returncode,
        "worktree_count": len(worktrees),
        "copied_chelated_dirs": copied_dirs,
        "snapshot_files": total_files,
        "snapshot_bytes": total_bytes,
    }


def stage_hardware_evidence() -> Dict[str, Any]:
    out_hw = STAGE_ROOT / "02-hardware-evidence"
    out_hw.mkdir(parents=True, exist_ok=True)

    # 1. Copy local STAGE0-NIGHT1 artifacts, docs/research, and archive vaults
    nf1, nb1 = _copy_tree_filtered(STAGE0_ROOT / "artifacts", out_hw / "bundles-27-of-27" / "artifacts")
    nf2, nb2 = _copy_tree_filtered(STAGE0_ROOT / "docs" / "research", out_hw / "bundles-27-of-27" / "docs-research")
    nf3, nb3 = _copy_tree_filtered(STAGE0_ROOT / "archive", out_hw / "local-archive-vaults")

    # 2. Pull remote DGX Spark `out/` manifests, receipts, and CSVs from gx10-37c3 (10.0.0.49) & gx10-6724 (10.0.0.194)
    for host_ip, node_name in (("10.0.0.49", "gx10-37c3"), ("10.0.0.194", "gx10-6724")):
        ndir = out_hw / "node-telemetry" / node_name
        ndir.mkdir(parents=True, exist_ok=True)
        tar_dest = ndir / f"{node_name}_chelated_night1_out_manifests_and_csvs.tar.gz"
        cmd = [
            "ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes", f"mattmre@{host_ip}",
            "cd /home/mattmre/chelated-night1 && tar -czf - out/ 2>/dev/null",
        ]
        try:
            with tar_dest.open("wb") as fh:
                subprocess.run(cmd, stdout=fh, timeout=180, check=False)
        except Exception:
            pass

    return {
        "local_evidence_files": nf1 + nf2 + nf3,
        "local_evidence_bytes": nb1 + nb2 + nb3,
    }


def stage_ai_provenance() -> Dict[str, Any]:
    out_ai = STAGE_ROOT / "03-ai-provenance"
    t3_out = out_ai / "t3-code-orchestrator"
    ag_out = out_ai / "antigravity-gemini-brain"
    cl_out = out_ai / "claude-code-sessions"
    cx_out = out_ai / "openai-codex-sessions"
    fp_out = out_ai / "frontier-panel-and-council"
    for p in (t3_out, ag_out, cl_out, cx_out, fp_out):
        p.mkdir(parents=True, exist_ok=True)

    # 1. T3 Code SQLite databases + CHELATEDAI thread exports + userdata\logs
    t3_ud = USER_HOME / ".t3" / "userdata"
    _safe_sqlite_backup(t3_ud / "statev2.sqlite", t3_out / "statev2.sqlite")
    _safe_sqlite_backup(t3_ud / "state.sqlite", t3_out / "state.sqlite")
    _copy_tree_filtered(t3_ud / "logs", t3_out / "logs")

    # Export CHELATEDAI-specific rows from statev2.sqlite as human-readable JSON if tables exist
    if (t3_out / "statev2.sqlite").is_file():
        try:
            con = sqlite3.connect(f"file:{t3_out / 'statev2.sqlite'}?mode=ro", uri=True)
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            export_summary: Dict[str, Any] = {"tables": tables, "chelated_matches": {}}
            for tbl in tables:
                try:
                    cols = [c[1] for c in con.execute(f"PRAGMA table_info({tbl})")]
                    rows = con.execute(f"SELECT * FROM {tbl}").fetchall()
                    matched = []
                    for r in rows:
                        s_row = json.dumps(r, default=str)
                        if "chelated" in s_row.lower() or "d68f68df" in s_row.lower():
                            matched.append(dict(zip(cols, [str(v)[:4000] if isinstance(v, (bytes, bytearray)) else v for v in r])))
                    if matched:
                        export_summary["chelated_matches"][tbl] = matched
                except Exception:
                    pass
            con.close()
            (t3_out / "thread-exports").mkdir(parents=True, exist_ok=True)
            (t3_out / "thread-exports" / "chelatedai_t3_statev2_export.json").write_text(
                json.dumps(export_summary, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    # 2. Antigravity / Gemini 4 Neon conversation brains inside .t3\userdata\providers\antigravity
    ag_prov = t3_ud / "providers" / "antigravity"
    ag_copied = 0
    if ag_prov.is_dir():
        for brain_root in ag_prov.glob("*/antigravity-acp/brain/*"):
            if not brain_root.is_dir():
                continue
            # Check if this brain belongs to CHELATEDAI (or current conversation / subagents)
            hit = brain_root.name in {
                "d68f68df-2385-4788-8983-e239ca3c27e0",
                "2a15c50b-d3c3-4bd3-9eed-538be41dd72a",
                "b4df008b-5f30-48bf-93c6-0d2a99b51c45",
                "d032e2fa-90fd-46c3-830b-57aa0f9f5241",
                "22c4aeb7-8475-4414-980c-d22dc6d4cb72",
                "496819f7-f6a6-45cf-a686-68c63d17c888",
            }
            if not hit:
                t_compact = brain_root / ".system_generated" / "logs" / "transcript.jsonl"
                if t_compact.is_file():
                    try:
                        chunk = t_compact.read_bytes()[:524288].lower()
                        if b"chelated" in chunk or b"ek5_v3" in chunk or b"spherical_migration" in chunk:
                            hit = True
                    except Exception:
                        pass
            if hit:
                _copy_tree_filtered(brain_root, ag_out / brain_root.name)
                ag_copied += 1

    # Also check other T3 providers (claude / codex / grok) under .t3\userdata\providers
    if (t3_ud / "providers").is_dir():
        for prov_dir in (t3_ud / "providers").iterdir():
            if prov_dir.is_dir() and prov_dir.name != "antigravity":
                _copy_tree_filtered(prov_dir, t3_out / "providers" / prov_dir.name)

    # 3. Claude Code `.claude\projects\*CHELATED*`
    cl_proj = USER_HOME / ".claude" / "projects"
    cl_copied = 0
    if cl_proj.is_dir():
        for d in cl_proj.iterdir():
            if d.is_dir() and "chelated" in d.name.lower():
                _copy_tree_filtered(d, cl_out / d.name)
                cl_copied += 1

    # 4. OpenAI Codex `.codex\sessions` & `archived_sessions` matching CHELATEDAI
    cx_root = USER_HOME / ".codex"
    cx_copied = 0
    for sub in ("sessions", "archived_sessions"):
        sdir = cx_root / sub
        if not sdir.is_dir():
            continue
        for jfile in sdir.rglob("*.jsonl"):
            try:
                head_bytes = jfile.read_bytes()[:262144].lower()
                if b"chelated" in head_bytes or b"ek5_v3" in head_bytes or b"spherical_migration" in head_bytes:
                    rel_j = jfile.relative_to(cx_root)
                    dest_j = cx_out / rel_j
                    dest_j.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(jfile, dest_j)
                    cx_copied += 1
            except Exception:
                pass

    # 5. Frontier Panel & Council logs
    _copy_tree_filtered(STAGE0_ROOT / "docs" / "research" / "council-logs", fp_out / "stage0-r5-panel-logs")
    if (GITHUB_ROOT / "CHELATEDAI" / "docs").is_dir():
        for f in (GITHUB_ROOT / "CHELATEDAI" / "docs").rglob("*council*"):
            if f.is_file():
                dest_f = fp_out / "root-repo-council" / f.name
                dest_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest_f)

    return {
        "antigravity_brains_copied": ag_copied,
        "claude_project_dirs_copied": cl_copied,
        "codex_session_files_copied": cx_copied,
    }


def main() -> None:
    t0 = time.perf_counter()
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    print("[VaultStage] 1. Staging Git bundle + 21 worktree snapshots...", flush=True)
    git_info = stage_git_and_worktrees()
    print(f"[VaultStage] Git & worktrees done: {git_info}", flush=True)

    print("[VaultStage] 2. Staging hardware evidence + remote DGX Spark telemetry...", flush=True)
    hw_info = stage_hardware_evidence()
    print(f"[VaultStage] Hardware evidence done: {hw_info}", flush=True)

    print("[VaultStage] 3. Staging multi-engine AI provenance (T3, Antigravity/Gemini, Claude, Codex, Grok/Panels)...", flush=True)
    ai_info = stage_ai_provenance()
    print(f"[VaultStage] AI provenance done: {ai_info}", flush=True)

    summary = {
        "vault_id": "CHELATEDAI-COMPLETE-VAULT-2026-11-22",
        "stage_wall_sec": round(time.perf_counter() - t0, 2),
        "git_and_worktrees": git_info,
        "hardware_evidence": hw_info,
        "ai_provenance": ai_info,
    }
    (STAGE_ROOT / "00-manifests").mkdir(parents=True, exist_ok=True)
    (STAGE_ROOT / "00-manifests" / "STAGE_PRELIM_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[VaultStage] ALL STAGING COMPLETE in {summary['stage_wall_sec']}s at {STAGE_ROOT}", flush=True)


if __name__ == "__main__":
    main()
