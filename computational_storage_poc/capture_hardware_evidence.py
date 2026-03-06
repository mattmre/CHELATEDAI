from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from payload_contract import TRIGGER_SECTOR_LBA, compute_payload_result
from usb_host_inference import read_inference_from_drive, resolve_drive_path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def classify_target(target: str) -> str:
    target_str = str(target)
    if os.path.exists(target_str):
        return "path"
    if target_str.isdigit():
        return "physical-drive-number"
    return "device-path"


def collect_windows_disk_metadata(target: str) -> dict[str, Any] | None:
    if os.name != "nt":
        return None
    target_str = str(target)
    if not target_str.isdigit():
        return None

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f"Get-Disk -Number {target_str} | "
            "Select-Object Number,FriendlyName,SerialNumber,BusType,PartitionStyle,"
            "OperationalStatus,HealthStatus,Size | ConvertTo-Json -Compress"
        ),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"metadata_error": str(exc)}

    output = result.stdout.strip()
    if not output:
        return None

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {"metadata_error": f"Unexpected PowerShell output: {output}"}

    if isinstance(parsed, dict):
        return parsed
    return {"metadata_error": f"Unexpected metadata shape: {type(parsed).__name__}"}


def compare_payloads(expected: dict[str, Any], observed: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if observed_value != expected_value:
            mismatches.append(
                {
                    "field": key,
                    "expected": expected_value,
                    "observed": observed_value,
                }
            )

    for key, observed_value in observed.items():
        if key not in expected:
            mismatches.append(
                {
                    "field": key,
                    "expected": "<missing>",
                    "observed": observed_value,
                }
            )

    return mismatches


def capture_hardware_evidence(
    target: str,
    *,
    notes: str = "",
    read_fn: Callable[..., str] = read_inference_from_drive,
    metadata_fn: Callable[[str], dict[str, Any] | None] = collect_windows_disk_metadata,
    now_fn: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    timestamp = now_fn()
    expected_result = compute_payload_result()
    observed_text = read_fn(target, sector_start=TRIGGER_SECTOR_LBA, num_sectors=1, verbose=False)
    observed_result = json.loads(observed_text)
    mismatches = compare_payloads(expected_result, observed_result)

    return {
        "captured_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "target": str(target),
        "target_kind": classify_target(target),
        "resolved_target_path": resolve_drive_path(target),
        "sector_lba": TRIGGER_SECTOR_LBA,
        "matches_expected_payload": not mismatches,
        "mismatches": mismatches,
        "expected_result": expected_result,
        "observed_result": observed_result,
        "notes": notes,
        "device_metadata": metadata_fn(target),
    }


def render_markdown_report(evidence: dict[str, Any]) -> str:
    status = "PASS" if evidence["matches_expected_payload"] else "FAIL"
    mismatches = evidence["mismatches"]
    mismatch_lines = ["- None"] if not mismatches else [
        f"- `{entry['field']}` expected `{entry['expected']}` observed `{entry['observed']}`"
        for entry in mismatches
    ]

    device_metadata = evidence.get("device_metadata")
    if device_metadata:
        metadata_block = json.dumps(device_metadata, indent=2, sort_keys=True)
    else:
        metadata_block = "null"

    return "\n".join(
        [
            "# RP2040 Hardware Evidence Report",
            "",
            f"- Status: **{status}**",
            f"- Captured at (UTC): `{evidence['captured_at_utc']}`",
            f"- Target: `{evidence['target']}`",
            f"- Target kind: `{evidence['target_kind']}`",
            f"- Resolved target path: `{evidence['resolved_target_path']}`",
            f"- Trigger sector: `{evidence['sector_lba']}`",
            f"- Platform: `{evidence['platform']}`",
            f"- Python: `{evidence['python_version']}`",
            "",
            "## Notes",
            evidence["notes"] or "- None",
            "",
            "## Mismatches",
            *mismatch_lines,
            "",
            "## Observed Result",
            "```json",
            json.dumps(evidence["observed_result"], indent=2, sort_keys=True),
            "```",
            "",
            "## Expected Result",
            "```json",
            json.dumps(evidence["expected_result"], indent=2, sort_keys=True),
            "```",
            "",
            "## Device Metadata",
            "```json",
            metadata_block,
            "```",
            "",
        ]
    )


def write_evidence_report(
    evidence: dict[str, Any],
    *,
    output_dir: Path,
    stem: str | None = None,
) -> tuple[Path, Path]:
    timestamp = evidence["captured_at_utc"].replace(":", "").replace("-", "").replace("Z", "Z")
    file_stem = stem or f"hardware-evidence-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{file_stem}.json"
    md_path = output_dir / f"{file_stem}.md"

    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(evidence) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a deterministic evidence report from the RP2040 transport path."
    )
    parser.add_argument(
        "target",
        help="Physical drive number, device path, or file-backed image path to read from.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/computational-storage-hardware-evidence",
        help="Directory where the JSON and Markdown reports should be written.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Optional output file stem. Defaults to a UTC timestamped stem.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional operator notes to embed in the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = capture_hardware_evidence(args.target, notes=args.notes)
    json_path, md_path = write_evidence_report(
        evidence,
        output_dir=Path(args.output_dir),
        stem=args.stem,
    )

    print(f"Wrote hardware evidence JSON report to {json_path}")
    print(f"Wrote hardware evidence Markdown report to {md_path}")
    print(f"Payload match status: {'PASS' if evidence['matches_expected_payload'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
