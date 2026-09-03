from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


if not __debug__:
    raise RuntimeError("optimized Python disables semantic checks; rerun without -O")


ROOT = Path(__file__).resolve().parent
EXACT_SOURCE_COMMIT = "8f1955abe8a214cc6f469547d3cc1bbddbf7d1ec"
ACCEPTANCE_SOURCE_COMMIT = "8669dcd77821d43a54601fffdcbd4668c8315e2f"
EXPECTED_SHA256 = {
    "primary-aggregate-80of80.json": "32370e4c85f52d1d6a77a718a41dff77b969582af8e0579894c07463b56b35b2",
    "freezer-gate-80of80.json": "36b86371819baa699eab2af20156892b8aaedf143ab2b92c0b2a31925db4419b",
    "r5-full-census.aggregate.json": "dfff9543251c58406f12e3aa49b493dd3c852823a7061ab2364f188a50027b95",
    "spark1-cpu-qdrant-smoke.json": "6163661009da9770d7cdcc1d786e3f9842c8e0e11cea6cb2900fdbdb50303c77",
    "spark2-cpu-qdrant-smoke.json": "c208be009d8d2c8c5fa0be1dd7dd7637e4ae942b30dc27268ff977dd87d87a0b",
    "spark2-focused-recovery-validation.json": "38a8a19dfcd3f8b16a7551a90d7d7728d8e4cc10fa6ae791549dd4c932d99b1e",
    "spark-dual-gpu-linux-acceptance.json": "6c2a87543c134f364a1da844bacf3156d3718f2a755efda4406a29999c88565d",
}
PUBLIC_CONTENT_FILES = (
    *EXPECTED_SHA256,
    "PRIMARY_CAMPAIGN_REPORT.md",
    "R5_FULL_CENSUS_REPORT.md",
    "validation.json",
)
FORBIDDEN_FIELDS = (
    b'"candidate_id"',
    b'"campaign_id"',
    b'"task_id"',
    b'"prompt_digest"',
    b'"original_response"',
    b'"task_statement"',
    b'"initial_source"',
)
FORBIDDEN_PATTERNS = {
    "ipv4_literal": re.compile(rb"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"),
    "endpoint_url": re.compile(rb"(?i)\b[a-z][a-z0-9+.-]*://"),
    "loopback_endpoint": re.compile(rb"(?i)\b(?:localhost|\[?::1\]?)\s*:\s*[0-9]{2,5}\b"),
    "unix_home_path": re.compile(rb"(?i)(?:^|[\\/])home[\\/][^\\/\r\n\"]+[\\/]"),
    "sensitive_unix_path": re.compile(
        rb"(?i)(?:^|[\s\"'`(:=])/(?:data|etc|home|mnt|opt|private|root|run|srv|tmp|users|var|workspace)(?:/|\\)"
    ),
    "windows_absolute_path": re.compile(rb"(?i)\b[a-z]:[\\/]+"),
    "windows_user_path": re.compile(
        rb"(?i)[a-z]:[\\/]+users[\\/]+[^\\/\r\n\"]+[\\/]"
    ),
    "unc_path": re.compile(rb"(?i)\\\\[a-z0-9._-]+[\\/]"),
    "private_host_label": re.compile(rb"(?i)\b(?:gx[0-9]+-[a-z0-9.-]+|(?:gba|eba)[0-9]+)\b"),
    "internal_dns_name": re.compile(
        rb"(?i)\b[a-z0-9_-]+(?:\.[a-z0-9_-]+)*\.(?:internal|lan|local)(?:\.[a-z0-9_-]+)*\b"
    ),
    "host_path_endpoint": re.compile(
        rb"(?i)\b[a-z][a-z0-9._-]{0,252}:(?:/|[a-z0-9._-]+/)"
    ),
    "scp_style_endpoint": re.compile(rb"(?i)\b[a-z0-9._-]+@[a-z0-9._-]+:"),
}
IPV6_CANDIDATE_RE = re.compile(rb"\[?([0-9a-fA-F:]*:[0-9a-fA-F:]+)\]?")
HOST_PORT_CANDIDATE_RE = re.compile(rb"\b([a-zA-Z][a-zA-Z0-9_-]{0,62}):([0-9]{2,5})\b")
ALLOWED_NON_ENDPOINT_HOST_PORT_PREFIXES = {
    b"A_STRICT_SINGLE_FENCE_PROJECTION",
    b"B_SOURCE_ONLY_PREFILL",
}
ARMS = {"B", "D"}
ATTEMPT_BINS = {"01-03", "04-06", "07-09", "10-12"}
FAMILIES = {
    "DATA_TRANSFORM",
    "DEPENDENCY_CONTRACT",
    "PARSER_EDGE",
    "PURE_FUNCTION",
    "RESOURCE_BOUND",
    "STATE_TRANSITION",
}
RETRIEVAL_BINS = {"01-03", "04-07", "08-15", "16+"}
SEEDS = {"0", "1"}
CONDITIONS = {"A_STRICT_SINGLE_FENCE_PROJECTION", "B_SOURCE_ONLY_PREFILL"}
DIAGNOSTICS = {"RUNTIME_EXCEPTION", "WRONG_OUTPUT"}
TIMING_KEYS = {"count", "max", "mean", "min", "total"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
UTC_RE = re.compile(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def require_keys(value: object, expected: set[str], locus: str) -> dict:
    if not isinstance(value, dict):
        raise AssertionError(f"{locus}: expected object")
    actual = set(value)
    if actual != expected:
        raise AssertionError(
            f"{locus} keys: actual={sorted(actual)} expected={sorted(expected)}"
        )
    return value


def require_int_map(value: object, expected: set[str], locus: str) -> dict:
    result = require_keys(value, expected, locus)
    for key, item in result.items():
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise AssertionError(f"{locus}.{key}: expected non-negative integer")
    return result


def require_bool_map(value: object, expected: set[str], locus: str) -> dict:
    result = require_keys(value, expected, locus)
    for key, item in result.items():
        if not isinstance(item, bool):
            raise AssertionError(f"{locus}.{key}: expected boolean")
    return result


def require_string_list(value: object, locus: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise AssertionError(f"{locus}: expected non-empty string list")
    return value


def require_sha256(value: object, locus: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AssertionError(f"{locus}: expected lowercase SHA-256")


def require_git_sha(value: object, locus: str) -> None:
    if not isinstance(value, str) or GIT_SHA_RE.fullmatch(value) is None:
        raise AssertionError(f"{locus}: expected lowercase full Git object ID")


def require_utc(value: object, locus: str) -> None:
    if not isinstance(value, str) or UTC_RE.fullmatch(value) is None:
        raise AssertionError(f"{locus}: expected UTC second timestamp")


def require_timing(value: object, locus: str) -> None:
    result = require_keys(value, TIMING_KEYS, locus)
    for key, item in result.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0:
            raise AssertionError(f"{locus}.{key}: expected non-negative number")


def scan_public_bytes(filename: str, raw: bytes) -> None:
    lowered = raw.lower()
    hits = [field.decode("ascii") for field in FORBIDDEN_FIELDS if field in lowered]
    hits.extend(name for name, pattern in FORBIDDEN_PATTERNS.items() if pattern.search(raw))
    for match in IPV6_CANDIDATE_RE.finditer(raw):
        try:
            ipaddress.IPv6Address(match.group(1).decode("ascii"))
        except ValueError:
            continue
        hits.append("ipv6_literal")
        break
    for match in HOST_PORT_CANDIDATE_RE.finditer(raw):
        if match.group(1) in ALLOWED_NON_ENDPOINT_HOST_PORT_PREFIXES:
            continue
        if 1 <= int(match.group(2)) <= 65535:
            hits.append("host_port_endpoint")
            break
    if hits:
        raise AssertionError(f"{filename}: public-safety scan hits {sorted(hits)}")


def validate_privacy_fault_fixtures() -> None:
    forbidden = (
        b"[fd00::24]:8443",
        b"evaluator.internal.example",
        b"/var/lib/private-evaluator/run.sock",
        b"operator@internal-evaluator:/srv/evidence",
        b"evaluator:8443",
        b"worker1:5000",
        b"EVALUATOR:8443",
        b"Worker1:5000",
        b"/Users/operator/private/run.sock",
        b"host:/relative/private",
    )
    for index, payload in enumerate(forbidden):
        try:
            scan_public_bytes(f"privacy-negative-{index}", payload)
        except AssertionError:
            continue
        raise AssertionError(f"privacy-negative-{index}: forbidden fixture was accepted")

    for index, payload in enumerate(
        (
            b"2026-08-24T20:17:57Z",
            b"docs/evidence/result.json",
            b"private fields are omitted",
            b"A_STRICT_SINGLE_FENCE_PROJECTION:01",
            b"B_SOURCE_ONLY_PREFILL:04",
        )
    ):
        scan_public_bytes(f"privacy-positive-{index}", payload)


def scan_public_content_files() -> None:
    for filename in PUBLIC_CONTENT_FILES:
        path = ROOT / filename
        if not path.is_file():
            raise AssertionError(f"{filename}: required public content file is missing")
        scan_public_bytes(filename, path.read_bytes())


def load_and_validate_bytes(filename: str) -> tuple[dict, str]:
    raw = (ROOT / filename).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise AssertionError(f"{filename}: expected a JSON object")
    if canonical(value) != raw:
        raise AssertionError(f"{filename}: bytes are not canonical JSON")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256[filename]:
        raise AssertionError(f"{filename}: digest mismatch")
    scan_public_bytes(filename, raw)
    return value, digest


def run_gitleaks() -> int:
    executable = shutil.which("gitleaks")
    if executable is None:
        raise AssertionError("gitleaks is required to reproduce this attestation")
    with tempfile.TemporaryDirectory(prefix="egv-public-evidence-") as directory:
        report = Path(directory) / "gitleaks.json"
        completed = subprocess.run(
            [
                executable,
                "dir",
                "--no-banner",
                "--no-color",
                "--redact=100",
                "--report-format",
                "json",
                "--report-path",
                str(report),
                "--exit-code",
                "2",
                str(ROOT),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode not in {0, 2}:
            raise AssertionError(
                f"gitleaks execution failed with exit code {completed.returncode}"
            )
        findings = json.loads(report.read_text(encoding="utf-8")) if report.exists() else []
        if not isinstance(findings, list):
            raise AssertionError("gitleaks report must be a JSON list")
        if completed.returncode == 0 and findings:
            raise AssertionError("gitleaks reported findings with a success exit code")
        if completed.returncode == 2 and not findings:
            raise AssertionError("gitleaks returned its findings code without findings")
        return len(findings)


primary, primary_digest = load_and_validate_bytes("primary-aggregate-80of80.json")
require_keys(
    primary,
    {
        "attempts",
        "evaluator",
        "evidence_use",
        "integrity_counters",
        "schema_version",
        "terminal",
    },
    "primary",
)
attempts = require_keys(
    primary["attempts"],
    {
        "by_arm",
        "by_seed",
        "by_task_family",
        "failure_family",
        "failure_family_by_arm",
        "failure_family_rate",
        "inflight_excluded",
        "position_status",
        "source_contract_failures_by_arm",
        "status",
        "status_by_arm",
        "status_by_seed",
        "status_by_task_family",
        "total",
    },
    "primary.attempts",
)
require_int_map(attempts["by_arm"], ARMS, "primary.attempts.by_arm")
require_int_map(attempts["by_seed"], SEEDS, "primary.attempts.by_seed")
require_int_map(attempts["by_task_family"], FAMILIES, "primary.attempts.by_task_family")
require_int_map(
    attempts["failure_family"],
    {"RESPONSE_CONTRACT:VariationDependencyError"},
    "primary.attempts.failure_family",
)
require_int_map(
    attempts["failure_family_by_arm"],
    {f"{arm}:RESPONSE_CONTRACT:VariationDependencyError" for arm in ARMS},
    "primary.attempts.failure_family_by_arm",
)
failure_rates = require_keys(
    attempts["failure_family_rate"],
    {"RESPONSE_CONTRACT:VariationDependencyError"},
    "primary.attempts.failure_family_rate",
)
if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in failure_rates.values()):
    raise AssertionError("primary.attempts.failure_family_rate: expected numeric values")
require_int_map(
    attempts["position_status"],
    {f"{position}:{status}" for position in range(1, 13) for status in ("FAILED", "SUCCESS")},
    "primary.attempts.position_status",
)
require_int_map(
    attempts["source_contract_failures_by_arm"],
    ARMS,
    "primary.attempts.source_contract_failures_by_arm",
)
require_int_map(attempts["status"], {"FAILED", "SUCCESS"}, "primary.attempts.status")
require_int_map(
    attempts["status_by_arm"],
    {f"{arm}:{status}" for arm in ARMS for status in ("FAILED", "SUCCESS")},
    "primary.attempts.status_by_arm",
)
require_int_map(
    attempts["status_by_seed"],
    {f"{seed}:{status}" for seed in SEEDS for status in ("FAILED", "SUCCESS")},
    "primary.attempts.status_by_seed",
)
require_int_map(
    attempts["status_by_task_family"],
    {f"{family}:{status}" for family in FAMILIES for status in ("FAILED", "SUCCESS")},
    "primary.attempts.status_by_task_family",
)
for scalar in ("inflight_excluded", "total"):
    if isinstance(attempts[scalar], bool) or not isinstance(attempts[scalar], int) or attempts[scalar] < 0:
        raise AssertionError(f"primary.attempts.{scalar}: expected non-negative integer")

evaluator = require_keys(
    primary["evaluator"],
    {"correctness", "diagnostics_by_arm", "diagnostics_by_seed", "diagnostics_by_task_family"},
    "primary.evaluator",
)
require_int_map(evaluator["correctness"], {"0"}, "primary.evaluator.correctness")
require_int_map(
    evaluator["diagnostics_by_arm"],
    {f"{arm}:{diagnostic}" for arm in ARMS for diagnostic in DIAGNOSTICS},
    "primary.evaluator.diagnostics_by_arm",
)
require_int_map(
    evaluator["diagnostics_by_seed"],
    {f"{seed}:{diagnostic}" for seed in SEEDS for diagnostic in DIAGNOSTICS},
    "primary.evaluator.diagnostics_by_seed",
)
require_int_map(
    evaluator["diagnostics_by_task_family"],
    {f"{family}:{family_diagnostic}" for family, family_diagnostic in {
        "DATA_TRANSFORM": "RUNTIME_EXCEPTION",
        "DEPENDENCY_CONTRACT": "WRONG_OUTPUT",
        "PARSER_EDGE": "WRONG_OUTPUT",
        "PURE_FUNCTION": "WRONG_OUTPUT",
        "RESOURCE_BOUND": "WRONG_OUTPUT",
        "STATE_TRANSITION": "WRONG_OUTPUT",
    }.items()},
    "primary.evaluator.diagnostics_by_task_family",
)

evidence_use = require_keys(primary["evidence_use"], ARMS, "primary.evidence_use")
for arm, values in evidence_use.items():
    arm_use = require_keys(
        values,
        {"evaluated_candidates", "opportunity_candidates", "use_candidates", "use_rate_when_available"},
        f"primary.evidence_use.{arm}",
    )
    for scalar in ("evaluated_candidates", "opportunity_candidates", "use_candidates"):
        if isinstance(arm_use[scalar], bool) or not isinstance(arm_use[scalar], int) or arm_use[scalar] < 0:
            raise AssertionError(f"primary.evidence_use.{arm}.{scalar}: expected non-negative integer")
    if isinstance(arm_use["use_rate_when_available"], bool) or not isinstance(
        arm_use["use_rate_when_available"], (int, float)
    ):
        raise AssertionError(f"primary.evidence_use.{arm}.use_rate_when_available: expected number")

integrity = require_int_map(
    primary["integrity_counters"],
    {
        "completed_candidates",
        "completed_checkpoints",
        "completed_effects",
        "completed_receipts",
        "live_candidates_including_inflight",
        "live_checkpoints_including_inflight",
        "live_effects_including_inflight",
        "live_receipts_including_inflight",
        "projection_queue",
        "quarantines",
    },
    "primary.integrity_counters",
)
terminal = require_keys(
    primary["terminal"],
    {"by_arm", "by_seed", "by_task_family", "completed", "promotions", "status"},
    "primary.terminal",
)
require_int_map(terminal["by_arm"], ARMS, "primary.terminal.by_arm")
require_int_map(terminal["by_seed"], SEEDS, "primary.terminal.by_seed")
require_int_map(terminal["by_task_family"], FAMILIES, "primary.terminal.by_task_family")
require_int_map(terminal["status"], {"BUDGET_EXHAUSTED"}, "primary.terminal.status")

assert primary["schema_version"] == "egv-primary-safe-aggregate-v1"
assert attempts["total"] == 960 and attempts["status"] == {"FAILED": 617, "SUCCESS": 343}
assert attempts["inflight_excluded"] == 0
assert sum(attempts["by_arm"].values()) == 960 == sum(attempts["by_seed"].values())
assert sum(attempts["by_task_family"].values()) == 960
assert attempts["source_contract_failures_by_arm"] == {"B": 306, "D": 311}
assert attempts["failure_family"] == {"RESPONSE_CONTRACT:VariationDependencyError": 617}
assert evaluator["correctness"] == {"0": 343}
assert sum(evaluator["diagnostics_by_arm"].values()) == 343
assert evidence_use["B"] == {
    "evaluated_candidates": 174,
    "opportunity_candidates": 154,
    "use_candidates": 154,
    "use_rate_when_available": 1.0,
}
assert evidence_use["D"] == {
    "evaluated_candidates": 169,
    "opportunity_candidates": 149,
    "use_candidates": 149,
    "use_rate_when_available": 1.0,
}
assert terminal["completed"] == 80 and terminal["promotions"] == 0
assert terminal["status"] == {"BUDGET_EXHAUSTED": 80}
assert integrity == {
    "completed_candidates": 343,
    "completed_checkpoints": 343,
    "completed_effects": 343,
    "completed_receipts": 1029,
    "live_candidates_including_inflight": 343,
    "live_checkpoints_including_inflight": 343,
    "live_effects_including_inflight": 343,
    "live_receipts_including_inflight": 1029,
    "projection_queue": 0,
    "quarantines": 0,
}

freezer, freezer_digest = load_and_validate_bytes("freezer-gate-80of80.json")
require_keys(
    freezer,
    {
        "candidate_count",
        "copied_ledger",
        "cutoff_digest",
        "dataset_digest",
        "distinct_task_count",
        "excluded_counts",
        "output_digest",
        "row_count",
        "schema",
        "source_commit",
        "training_gate",
    },
    "freezer",
)
require_bool_map(freezer["copied_ledger"], {"integrity", "unchanged"}, "freezer.copied_ledger")
require_int_map(
    freezer["excluded_counts"],
    {"invalid_or_ineligible_attempt"},
    "freezer.excluded_counts",
)
require_bool_map(
    freezer["training_gate"],
    {
        "accepted_pre_model_rejection",
        "adapter_created",
        "hardened_overlay_pending",
        "pre_model_no_model_claimed",
        "training_output_created",
        "zero_row_rejection_observed_on_exact_source",
    },
    "freezer.training_gate",
)
for field in ("cutoff_digest", "dataset_digest", "output_digest"):
    require_sha256(freezer[field], f"freezer.{field}")
require_git_sha(freezer["source_commit"], "freezer.source_commit")
for scalar in ("candidate_count", "distinct_task_count", "row_count"):
    if isinstance(freezer[scalar], bool) or not isinstance(freezer[scalar], int) or freezer[scalar] < 0:
        raise AssertionError(f"freezer.{scalar}: expected non-negative integer")
assert freezer["schema"] == "egv.public-safe.freezer-gate.v1"
assert freezer["candidate_count"] == 343
assert freezer["excluded_counts"] == {"invalid_or_ineligible_attempt": 343}
assert freezer["row_count"] == 0 and freezer["distinct_task_count"] == 0
assert freezer["copied_ledger"] == {"integrity": True, "unchanged": True}
assert freezer["training_gate"] == {
    "accepted_pre_model_rejection": False,
    "adapter_created": False,
    "hardened_overlay_pending": True,
    "pre_model_no_model_claimed": False,
    "training_output_created": False,
    "zero_row_rejection_observed_on_exact_source": True,
}


r5, r5_digest = load_and_validate_bytes("r5-full-census.aggregate.json")
require_keys(
    r5,
    {
        "claim_bounds",
        "frozen_failure_census",
        "generation_diagnostic",
        "isolation",
        "model",
        "schema",
        "semantic_diagnostic",
        "source_commit",
        "upstream_bindings",
    },
    "r5",
)
require_string_list(r5["claim_bounds"], "r5.claim_bounds")
frozen = require_keys(
    r5["frozen_failure_census"],
    {
        "arm_counts",
        "attempt_bins",
        "family_counts",
        "matched_contexts",
        "primary_terminal_cutoff",
        "retrieval_bins",
        "seed_counts",
        "selection",
    },
    "r5.frozen_failure_census",
)
require_int_map(frozen["arm_counts"], ARMS, "r5.frozen_failure_census.arm_counts")
require_int_map(frozen["attempt_bins"], ATTEMPT_BINS, "r5.frozen_failure_census.attempt_bins")
require_int_map(frozen["family_counts"], FAMILIES, "r5.frozen_failure_census.family_counts")
require_int_map(frozen["retrieval_bins"], RETRIEVAL_BINS, "r5.frozen_failure_census.retrieval_bins")
require_int_map(frozen["seed_counts"], SEEDS, "r5.frozen_failure_census.seed_counts")
if not isinstance(frozen["selection"], str):
    raise AssertionError("r5.frozen_failure_census.selection: expected string")

generation = require_keys(
    r5["generation_diagnostic"],
    {
        "B_production_valid",
        "B_source_contract_accepted",
        "completed_generations",
        "conditions",
        "paired_source_digest_different",
        "paired_source_digest_equal",
        "source_digest_uniqueness",
        "timing_ms",
        "wall_time_ms",
    },
    "r5.generation_diagnostic",
)
conditions = require_keys(generation["conditions"], {"A", "B"}, "r5.generation.conditions")
if not all(isinstance(value, str) for value in conditions.values()):
    raise AssertionError("r5.generation.conditions: expected string values")
uniqueness = require_keys(
    generation["source_digest_uniqueness"],
    {"A_unique", "B_unique", "by_family", "intersection", "union"},
    "r5.generation.source_digest_uniqueness",
)
family_uniqueness = require_keys(
    uniqueness["by_family"], FAMILIES, "r5.generation.source_digest_uniqueness.by_family"
)
for family, values in family_uniqueness.items():
    require_int_map(
        values,
        {"A_unique", "B_unique", "context_count", "intersection", "union"},
        f"r5.generation.source_digest_uniqueness.by_family.{family}",
    )
require_timing(generation["timing_ms"], "r5.generation.timing_ms")

require_bool_map(
    r5["isolation"],
    {
        "fresh_signer",
        "fresh_state",
        "generator_clean_exit",
        "generator_hidden_access",
        "primary_immutable_artifacts_unchanged",
        "primary_ledger_or_cache_writes",
        "read_only_copied_hidden_seed",
    },
    "r5.isolation",
)
model = require_keys(r5["model"], {"manifest_digest", "name", "revision"}, "r5.model")
require_sha256(model["manifest_digest"], "r5.model.manifest_digest")
if not all(isinstance(model[key], str) for key in ("name", "revision")):
    raise AssertionError("r5.model: expected string name and revision")

semantic = require_keys(
    r5["semantic_diagnostic"],
    {
        "completed_evaluations",
        "diagnostics",
        "diagnostics_by_arm",
        "diagnostics_by_attempt_bin",
        "diagnostics_by_family",
        "diagnostics_by_retrieval_bin",
        "diagnostics_by_seed",
        "infrastructure_loss_count",
        "paired_agreement_transitions",
        "pass_candidates",
        "timing_ms",
        "wall_time_ms",
    },
    "r5.semantic_diagnostic",
)
require_int_map(
    semantic["diagnostics"],
    {f"{condition}:{diagnostic}" for condition in CONDITIONS for diagnostic in DIAGNOSTICS},
    "r5.semantic.diagnostics",
)
require_int_map(
    semantic["diagnostics_by_arm"],
    {f"{condition}:{arm}:{diagnostic}" for condition in CONDITIONS for arm in ARMS for diagnostic in DIAGNOSTICS},
    "r5.semantic.diagnostics_by_arm",
)
require_int_map(
    semantic["diagnostics_by_attempt_bin"],
    {f"{condition}:{bucket}:{diagnostic}" for condition in CONDITIONS for bucket in ATTEMPT_BINS for diagnostic in DIAGNOSTICS},
    "r5.semantic.diagnostics_by_attempt_bin",
)
family_diagnostic = {
    "DATA_TRANSFORM": "RUNTIME_EXCEPTION",
    "DEPENDENCY_CONTRACT": "WRONG_OUTPUT",
    "PARSER_EDGE": "WRONG_OUTPUT",
    "PURE_FUNCTION": "WRONG_OUTPUT",
    "RESOURCE_BOUND": "WRONG_OUTPUT",
    "STATE_TRANSITION": "WRONG_OUTPUT",
}
require_int_map(
    semantic["diagnostics_by_family"],
    {f"{condition}:{family}:{diagnostic}" for condition in CONDITIONS for family, diagnostic in family_diagnostic.items()},
    "r5.semantic.diagnostics_by_family",
)
retrieval_diagnostics = {
    "01-03": DIAGNOSTICS,
    "04-07": DIAGNOSTICS,
    "08-15": {"WRONG_OUTPUT"},
    "16+": {"WRONG_OUTPUT"},
}
require_int_map(
    semantic["diagnostics_by_retrieval_bin"],
    {f"{condition}:{bucket}:{diagnostic}" for condition in CONDITIONS for bucket, diagnostics in retrieval_diagnostics.items() for diagnostic in diagnostics},
    "r5.semantic.diagnostics_by_retrieval_bin",
)
require_int_map(
    semantic["diagnostics_by_seed"],
    {f"{condition}:{seed}:{diagnostic}" for condition in CONDITIONS for seed in SEEDS for diagnostic in DIAGNOSTICS},
    "r5.semantic.diagnostics_by_seed",
)
require_int_map(
    semantic["paired_agreement_transitions"],
    {
        "FAIL_FAIL",
        "RUNTIME_EXCEPTION->RUNTIME_EXCEPTION",
        "WRONG_OUTPUT->WRONG_OUTPUT",
        "exact_diagnostic_agreement",
        "pass_fail_agreement",
    },
    "r5.semantic.paired_agreement_transitions",
)
require_int_map(semantic["pass_candidates"], {"A", "B", "total"}, "r5.semantic.pass_candidates")
semantic_timing = require_keys(semantic["timing_ms"], CONDITIONS, "r5.semantic.timing_ms")
for condition, timing in semantic_timing.items():
    require_timing(timing, f"r5.semantic.timing_ms.{condition}")

upstream = require_keys(
    r5["upstream_bindings"],
    {
        "evaluation_preregistration_digest",
        "evaluation_results_digest",
        "evaluation_service_manifest_digest",
        "generation_preregistration_digest",
        "generation_results_digest",
        "generation_seal_digest",
    },
    "r5.upstream_bindings",
)
for key, value in upstream.items():
    require_sha256(value, f"r5.upstream_bindings.{key}")

assert r5["schema"] == "egv.public-safe.r5-full-census.v1"
assert r5["frozen_failure_census"]["matched_contexts"] == 387
assert r5["generation_diagnostic"]["completed_generations"] == 387
assert r5["generation_diagnostic"]["B_source_contract_accepted"] == 387
assert r5["generation_diagnostic"]["B_production_valid"] == 387
assert r5["generation_diagnostic"]["paired_source_digest_equal"] == 0
assert r5["generation_diagnostic"]["paired_source_digest_different"] == 387
assert r5["semantic_diagnostic"]["completed_evaluations"] == 774
assert r5["semantic_diagnostic"]["pass_candidates"] == {"A": 0, "B": 0, "total": 0}
assert r5["semantic_diagnostic"]["infrastructure_loss_count"] == 0
assert r5["semantic_diagnostic"]["paired_agreement_transitions"]["FAIL_FAIL"] == 387
assert r5["isolation"] == {
    "fresh_signer": True,
    "fresh_state": True,
    "generator_clean_exit": True,
    "generator_hidden_access": False,
    "primary_immutable_artifacts_unchanged": True,
    "primary_ledger_or_cache_writes": False,
    "read_only_copied_hidden_seed": True,
}

smoke, smoke_digest = load_and_validate_bytes("spark2-cpu-qdrant-smoke.json")
require_keys(
    smoke,
    {
        "completed_utc",
        "dependency_freeze_sha256",
        "dependency_multiset_matches_reference",
        "dependency_reference_sha256",
        "environment_archive_sha256",
        "python",
        "qdrant_client_version",
        "qdrant_tests",
        "runtime_generated_file_count",
        "schema",
        "scope",
        "smoke",
        "source_archive_sha256",
        "source_commit",
        "source_file_count",
        "source_preexecution_manifest_sha256",
        "started_utc",
    },
    "smoke",
)
require_utc(smoke["started_utc"], "smoke.started_utc")
require_utc(smoke["completed_utc"], "smoke.completed_utc")
require_string_list(smoke["scope"], "smoke.scope")
for key in (
    "dependency_freeze_sha256",
    "dependency_reference_sha256",
    "environment_archive_sha256",
    "source_archive_sha256",
    "source_preexecution_manifest_sha256",
):
    require_sha256(smoke[key], f"smoke.{key}")
qdrant = require_keys(
    smoke["qdrant_tests"],
    {"exit_code", "passed", "reported_run_count", "selected_count", "stderr_sha256", "stdout_sha256"},
    "smoke.qdrant_tests",
)
require_sha256(qdrant["stderr_sha256"], "smoke.qdrant_tests.stderr_sha256")
require_sha256(qdrant["stdout_sha256"], "smoke.qdrant_tests.stdout_sha256")
smoke_run = require_keys(
    smoke["smoke"],
    {"command_kind", "exit_code", "json_parseable", "json_top_level_type", "passed", "stderr_sha256", "stdout_sha256"},
    "smoke.smoke",
)
require_sha256(smoke_run["stderr_sha256"], "smoke.smoke.stderr_sha256")
require_sha256(smoke_run["stdout_sha256"], "smoke.smoke.stdout_sha256")
assert smoke["schema"] == "egv.public-safe.cpu-qdrant-smoke.v1"
assert smoke["source_file_count"] == 929
assert smoke["dependency_multiset_matches_reference"] is True
assert smoke["qdrant_client_version"] == "1.17.1"
assert smoke["smoke"]["exit_code"] == 0 and smoke["smoke"]["passed"] is True
assert smoke["smoke"]["json_parseable"] is True
assert smoke["qdrant_tests"]["selected_count"] == 2
assert smoke["qdrant_tests"]["reported_run_count"] == 2
assert smoke["qdrant_tests"]["exit_code"] == 0 and smoke["qdrant_tests"]["passed"] is True

spark1_smoke, spark1_smoke_digest = load_and_validate_bytes("spark1-cpu-qdrant-smoke.json")
require_keys(
    spark1_smoke,
    {
        "completed_utc",
        "dependency_freeze_sha256",
        "dependency_multiset_matches_reference",
        "dependency_reference_sha256",
        "environment_archive_sha256",
        "python",
        "qdrant_client_version",
        "qdrant_tests",
        "runtime_generated_file_count",
        "schema",
        "scope",
        "smoke",
        "source_archive_sha256",
        "source_commit",
        "source_file_count",
        "source_preexecution_manifest_sha256",
        "started_utc",
    },
    "spark1_smoke",
)
require_utc(spark1_smoke["started_utc"], "spark1_smoke.started_utc")
require_utc(spark1_smoke["completed_utc"], "spark1_smoke.completed_utc")
require_git_sha(spark1_smoke["source_commit"], "spark1_smoke.source_commit")
require_string_list(spark1_smoke["scope"], "spark1_smoke.scope")
for key in (
    "dependency_freeze_sha256",
    "dependency_reference_sha256",
    "environment_archive_sha256",
    "source_archive_sha256",
    "source_preexecution_manifest_sha256",
):
    require_sha256(spark1_smoke[key], f"spark1_smoke.{key}")
spark1_qdrant = require_keys(
    spark1_smoke["qdrant_tests"],
    {"exit_code", "passed", "reported_run_count", "selected_count", "stderr_sha256", "stdout_sha256"},
    "spark1_smoke.qdrant_tests",
)
require_sha256(spark1_qdrant["stderr_sha256"], "spark1_smoke.qdrant_tests.stderr_sha256")
require_sha256(spark1_qdrant["stdout_sha256"], "spark1_smoke.qdrant_tests.stdout_sha256")
spark1_run = require_keys(
    spark1_smoke["smoke"],
    {
        "command_kind",
        "exit_code",
        "json_parseable",
        "json_top_level_type",
        "passed",
        "stderr_sha256",
        "stdout_sha256",
    },
    "spark1_smoke.smoke",
)
require_sha256(spark1_run["stderr_sha256"], "spark1_smoke.smoke.stderr_sha256")
require_sha256(spark1_run["stdout_sha256"], "spark1_smoke.smoke.stdout_sha256")
assert spark1_smoke["schema"] == "egv.public-safe.cpu-qdrant-smoke.v1"
assert spark1_smoke["source_file_count"] == 929
assert spark1_smoke["runtime_generated_file_count"] == 0
assert spark1_smoke["dependency_multiset_matches_reference"] is True
assert spark1_smoke["qdrant_client_version"] == "1.17.1"
assert spark1_run["exit_code"] == 0 and spark1_run["passed"] is True
assert spark1_run["json_parseable"] is True
assert spark1_qdrant["selected_count"] == 2
assert spark1_qdrant["reported_run_count"] == 2
assert spark1_qdrant["exit_code"] == 0 and spark1_qdrant["passed"] is True

focused, focused_digest = load_and_validate_bytes("spark2-focused-recovery-validation.json")
require_keys(
    focused,
    {
        "completed_utc",
        "exit_code",
        "passed",
        "reported_skip_count",
        "reported_test_count",
        "schema",
        "scope",
        "selected_modules",
        "source_archive_sha256",
        "source_commit",
        "started_utc",
        "stderr_sha256",
        "stdout_sha256",
    },
    "focused",
)
require_utc(focused["started_utc"], "focused.started_utc")
require_utc(focused["completed_utc"], "focused.completed_utc")
require_string_list(focused["scope"], "focused.scope")
require_string_list(focused["selected_modules"], "focused.selected_modules")
for key in ("source_archive_sha256", "stderr_sha256", "stdout_sha256"):
    require_sha256(focused[key], f"focused.{key}")
assert focused["schema"] == "egv.public-safe.spark2-focused-recovery-validation.v1"
assert focused["reported_test_count"] == 157
assert focused["reported_skip_count"] == 0
assert focused["exit_code"] == 0 and focused["passed"] is True
assert len(focused["selected_modules"]) == 6

dual, dual_digest = load_and_validate_bytes("spark-dual-gpu-linux-acceptance.json")
require_keys(
    dual,
    {
        "accelerators",
        "claim_bounds",
        "genuine_zero_row",
        "recorded_utc",
        "schema",
        "selected_tests",
        "source_archive_sha256",
        "source_commit",
    },
    "dual",
)
require_utc(dual["recorded_utc"], "dual.recorded_utc")
require_git_sha(dual["source_commit"], "dual.source_commit")
require_sha256(dual["source_archive_sha256"], "dual.source_archive_sha256")
require_string_list(dual["claim_bounds"], "dual.claim_bounds")
require_string_list(dual["selected_tests"], "dual.selected_tests")
if not isinstance(dual["accelerators"], list) or len(dual["accelerators"]) != 2:
    raise AssertionError("dual.accelerators: expected two records")
for index, value in enumerate(dual["accelerators"], 1):
    accelerator = require_keys(
        value,
        {"accelerator_index", "image_digest", "implementation_acceptance", "runtime"},
        f"dual.accelerators.{index}",
    )
    require_sha256(accelerator["image_digest"], f"dual.accelerators.{index}.image_digest")
    acceptance = require_keys(
        accelerator["implementation_acceptance"],
        {
            "exit_code",
            "network_disabled",
            "passed",
            "read_only_source",
            "reported_skip_count",
            "reported_test_count",
        },
        f"dual.accelerators.{index}.implementation_acceptance",
    )
    runtime = require_keys(
        accelerator["runtime"],
        {"cuda_available", "device", "peft", "python", "torch", "transformers"},
        f"dual.accelerators.{index}.runtime",
    )
    assert accelerator["accelerator_index"] == index
    assert acceptance == {
        "exit_code": 0,
        "network_disabled": True,
        "passed": True,
        "read_only_source": True,
        "reported_skip_count": 0,
        "reported_test_count": 6,
    }
    assert runtime == {
        "cuda_available": True,
        "device": "NVIDIA GB10",
        "peft": "0.20.0",
        "python": "3.12.13",
        "torch": "2.11.0+cu130",
        "transformers": "5.13.1",
    }

zero_row = require_keys(
    dual["genuine_zero_row"],
    {
        "artifact_sha256",
        "candidate_count",
        "dataset_digest",
        "distinct_task_count",
        "expected_error",
        "per_accelerator",
        "row_count",
        "status",
    },
    "dual.genuine_zero_row",
)
require_sha256(zero_row["artifact_sha256"], "dual.genuine_zero_row.artifact_sha256")
require_sha256(zero_row["dataset_digest"], "dual.genuine_zero_row.dataset_digest")
if not isinstance(zero_row["per_accelerator"], list) or len(zero_row["per_accelerator"]) != 2:
    raise AssertionError("dual.genuine_zero_row.per_accelerator: expected two records")
for index, value in enumerate(zero_row["per_accelerator"], 1):
    proof = require_keys(
        value,
        {
            "accelerator_index",
            "adapter_output_absent_after",
            "adapter_output_absent_before",
            "exit_code",
            "gpu_exposed",
            "missing_dependency_paths_unresolved",
            "passed",
        },
        f"dual.genuine_zero_row.per_accelerator.{index}",
    )
    assert proof == {
        "accelerator_index": index,
        "adapter_output_absent_after": True,
        "adapter_output_absent_before": True,
        "exit_code": 1,
        "gpu_exposed": True,
        "missing_dependency_paths_unresolved": True,
        "passed": True,
    }
assert dual["schema"] == "egv.public-safe.dual-gpu-linux-acceptance.v1"
assert dual["source_commit"] == ACCEPTANCE_SOURCE_COMMIT
assert len(dual["selected_tests"]) == 6
assert zero_row["artifact_sha256"] == freezer["output_digest"]
assert zero_row["dataset_digest"] == freezer["dataset_digest"]
assert zero_row["candidate_count"] == 343
assert zero_row["distinct_task_count"] == 0
assert zero_row["row_count"] == 0
assert zero_row["status"] == "NO_ADMISSIBLE_TRAINING_SET"
assert zero_row["expected_error"] == "production selection requires exactly 20 frozen training tasks"

assert {
    freezer["source_commit"],
    r5["source_commit"],
    spark1_smoke["source_commit"],
    smoke["source_commit"],
    focused["source_commit"],
} == {EXACT_SOURCE_COMMIT}
assert (
    spark1_smoke["source_archive_sha256"]
    == smoke["source_archive_sha256"]
    == focused["source_archive_sha256"]
)
assert spark1_smoke["environment_archive_sha256"] == smoke["environment_archive_sha256"]
assert spark1_smoke["dependency_reference_sha256"] == smoke["dependency_reference_sha256"]
assert spark1_smoke["dependency_freeze_sha256"] == smoke["dependency_freeze_sha256"]
assert spark1_smoke["qdrant_client_version"] == smoke["qdrant_client_version"]

validate_privacy_fault_fixtures()
scan_public_content_files()
gitleaks_findings = run_gitleaks()
if gitleaks_findings:
    raise AssertionError(f"gitleaks found {gitleaks_findings} potential secrets")

result = {
    "schema": "egv.public-safe.artifact-validation.v1",
    "artifacts": {
        "primary-aggregate-80of80.json": primary_digest,
        "freezer-gate-80of80.json": freezer_digest,
        "r5-full-census.aggregate.json": r5_digest,
        "spark1-cpu-qdrant-smoke.json": spark1_smoke_digest,
        "spark2-cpu-qdrant-smoke.json": smoke_digest,
        "spark2-focused-recovery-validation.json": focused_digest,
        "spark-dual-gpu-linux-acceptance.json": dual_digest,
    },
    "canonical_json": True,
    "closed_schema": True,
    "exact_artifact_bytes_pinned": True,
    "gitleaks_findings": gitleaks_findings,
    "privacy_scan_forbidden_hits": [],
    "raw_prompt_or_source_fields_emitted": False,
    "private_identifier_fields_emitted": False,
    "status": "PASS",
}
result_bytes = canonical(result)
if (ROOT / "validation.json").read_bytes() != result_bytes:
    raise AssertionError("validation.json does not exactly reproduce validator output")
print(result_bytes.decode("ascii"), end="")
