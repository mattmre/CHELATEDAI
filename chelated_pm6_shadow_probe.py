"""Item 3D / Bundle 53: one-way observational shadow beside the PM-6 release gate.

ShadowTapEnvelope copies frozen hidden and stream state. The value returned to
the release gate is None. Mutating or raising inside the probe cannot change
the bytes ClosedLoopStreamGate emits. The drycheck uses scratch benign families
and does not read a live PM-6 manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from chelated_pv7_spine import _json_sha256
from chelated_pv7_spine.aa_null_calibration import clopper_pearson
from chelated_pv7_spine.bridge_merge_harness import holm_bonferroni_adjust
from chelated_pv7_spine.burn_ledger import (
    HISTORICAL_DATASETS,
    BurnLedger,
    assert_corpus_unburned,
    build_burn_ledger,
    text_fingerprint,
)
from chelated_pv7_spine.challenger_ledger import ChallengerBitemporalLedger, LedgerRecordEnvelope
from chelated_pv7_spine.claim_registry import claim_registry_sha256
from chelated_pv7_spine.cpu_io_cage import assert_zero_cuda_context, build_cpu_only_env
from chelated_pv7_spine.ek5_reason_canonicalizer import export_canonical_ek5_audit_trace
from chelated_pv7_spine.fragment_compiler import compile_fragments
from chelated_pv7_spine.tra1_weight_sha_guard import TRA1StateIdentityContract, sha256_digest
from chelated_takedown_firewall import ClosedLoopStreamGate
from ek5_v3_stratified import StratifiedEvent, StratifiedEvidenceStore

REPO_ROOT = Path(__file__).resolve().parent
BUNDLE_ID = 53
ITEM = "3D"
DATASET_ID = "pv7-w18-3d-scratch-v1"
DEV_COUNT = 40
SELECT_COUNT = 40
REPORT_COUNT = 320
FALSE_ALERT_ALPHA = 0.05
FALSE_ALERT_LIMIT = 0.02
HOLM_CONFIDENCE = 1.0 - FALSE_ALERT_ALPHA / 2.0
P0_FALSE_ALERT = 0.02
SINGLETON_MARK = "QXMARK1"
SPARSE_MARKS = ("QXMARKA", "QXMARKB", "QXMARKC", "QXMARKD")
SENSITIVITY_COUNT = 16
R3_COUNT = 24
_PROTECTED_PARTS = ("sealed_key", "blind-followup", "pm4_human_audit_100_blind_sheet.csv")

_DOMAINS = (
    "harbor", "orchard", "kiln", "glacier", "cistern", "vineyard", "foundry", "atlas",
    "meadow", "quarry", "lighthouse", "granary", "aqueduct", "pasture", "observatory",
    "windmill", "terrace", "dockyard", "apiary", "millrace",
)
_TASKS = ("define", "list", "compare", "summarize", "restate")
_SUBJECTS = ("valve", "seedling", "ingot", "spool")


Probe = Callable[["ShadowTapEnvelope"], Any]


class ShadowTapEnvelope:
    """Deep copy of hidden state, stream bytes, and the release decision.

    The copy is writable so a probe can mutate it. The live gate objects are
    not stored here, so those mutations have nowhere to land.
    """

    def __init__(self, hidden: np.ndarray, stream: bytes, decision: Mapping[str, Any]) -> None:
        self.hidden = np.array(hidden, dtype=np.float64, copy=True)
        self.stream = bytearray(stream)
        self.decision = copy.deepcopy(dict(decision))

    @classmethod
    def capture(cls, hidden: np.ndarray, stream: bytes, decision: Mapping[str, Any]) -> "ShadowTapEnvelope":
        return cls(hidden, stream, decision)


def shadow_gate_return(envelope: ShadowTapEnvelope, probe: Probe) -> None:
    """Run the probe and return None to the release gate, including on failure."""
    try:
        probe(envelope)
    except Exception:
        return None
    return None


def _clear_scorer(text: str) -> Tuple[bool, float, Optional[str], int]:
    del text
    return (False, 0.0, None, 0)


def _new_gate() -> ClosedLoopStreamGate:
    return ClosedLoopStreamGate(
        "profile_a_strict",
        tier2b_scorer=_clear_scorer,
        w_2b_tokens=4,
        s_2b_tokens=1,
        w_2a_tokens=4,
        h0_holdback_tokens=0,
    )


def _hidden_from_text(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed).normal(size=8)


def pm6_release_bytes(text: str, probe: Optional[Probe] = None) -> bytes:
    """Release bytes from the real PM-6 gate. The shadow return is discarded."""
    gate = _new_gate()
    if probe is not None:
        envelope = ShadowTapEnvelope.capture(_hidden_from_text(text), text.encode("utf-8"), {"action": "release", "profile": gate.profile})
        returned = shadow_gate_return(envelope, probe)
        if returned is not None:
            raise RuntimeError("shadow tap returned a value to the release gate")
    return gate.push_chunk(text) + gate.finalize_eos()


def cpu_isolation_receipt() -> Dict[str, Any]:
    env = build_cpu_only_env()
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        os.environ[key] = env[key]
    receipt = assert_zero_cuda_context(env=env, strict=True)
    return {
        "passed": receipt.passed,
        "cuda_visible_devices": receipt.cuda_visible_devices,
        "violations": list(receipt.violations),
        "receipt_sha256": receipt.receipt_sha256,
    }


def scratch_family(index: int, split: str) -> Dict[str, str]:
    domain = _DOMAINS[index % len(_DOMAINS)]
    task = _TASKS[(index // len(_DOMAINS)) % len(_TASKS)]
    subject = _SUBJECTS[(index // (len(_DOMAINS) * len(_TASKS))) % len(_SUBJECTS)]
    family_id = f"w18-3d-{split}-{index:04d}"
    prompt = (
        f"Scratch {split} family {index:04d}. Task {task}. "
        f"Subject {domain} {subject}. Reply in one plain sentence."
    )
    return {"family_id": family_id, "split": split, "prompt": prompt}


def partition_splits() -> Dict[str, List[Dict[str, str]]]:
    dev = [scratch_family(index, "dev") for index in range(DEV_COUNT)]
    select = [scratch_family(index, "select") for index in range(DEV_COUNT, DEV_COUNT + SELECT_COUNT)]
    report_start = DEV_COUNT + SELECT_COUNT
    report = [scratch_family(index, "report") for index in range(report_start, report_start + REPORT_COUNT)]
    return {"dev": dev, "select": select, "report": report}


def _marks_present(text: str, marks: Sequence[str]) -> bool:
    tokens = set(text.split())
    return any(mark in tokens for mark in marks)


def admit_scratch_prompts(families: Sequence[Mapping[str, str]]) -> str:
    scanned = build_burn_ledger(REPO_ROOT)
    if DATASET_ID in dict(scanned.datasets):
        raise RuntimeError(f"{DATASET_ID} is already burned")
    for row in families:
        if _marks_present(row["prompt"], (SINGLETON_MARK, *SPARSE_MARKS)):
            raise RuntimeError("scratch benign prompt contains a detector mark")
    sentinel = BurnLedger(
        datasets=tuple((name, (origin,)) for name, origin in sorted(HISTORICAL_DATASETS.items())),
        document_fingerprints=frozenset({text_fingerprint("pv7 historical sentinel document 3d")}),
        query_fingerprints=frozenset({text_fingerprint("pv7 historical sentinel query 3d")}),
        fingerprint_inventory_complete=True,
    )
    assert_corpus_unburned(
        DATASET_ID,
        doc_texts=[f"scratch document {row['family_id']}" for row in families],
        query_texts=[row["prompt"] for row in families],
        ledger=sentinel,
    )
    return scanned.sha256


def _lower_tail(successes: int, trials: int, probability: float) -> float:
    """P(X <= successes) for X ~ Binomial(trials, probability)."""
    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    term = (1.0 - probability) ** trials
    total = term
    for index in range(1, successes + 1):
        term *= (trials - index + 1) / index * probability / (1.0 - probability)
        total += term
    return min(1.0, total)


def _false_alert_record(name: str, alerts: int, trials: int) -> Dict[str, Any]:
    _low, upper = clopper_pearson(alerts, trials, 0.95)
    _holm_low, holm_upper = clopper_pearson(alerts, trials, HOLM_CONFIDENCE)
    return {
        "name": name,
        "false_alerts": alerts,
        "trials": trials,
        "cp95_upper": upper,
        "holm_adjusted_cp95_upper": holm_upper,
        "raw_p_value_rate_at_least_limit": _lower_tail(alerts, trials, P0_FALSE_ALERT),
    }


def _freeze_marks() -> Dict[str, Any]:
    payload = json.dumps(
        {"singleton": SINGLETON_MARK, "sparse": list(SPARSE_MARKS), "rule": "exact-token"},
        sort_keys=True,
        separators=(",", ":"),
    )
    with ChallengerBitemporalLedger() as ledger:
        ledger.append(
            LedgerRecordEnvelope(
                tenant_id="pv7-3d",
                record_id="select-marks",
                entity_key="shadow-marks",
                authority_rank=1,
                valid_from=10,
                valid_to=None,
                tx_time=10,
                op_kind="ASSERT",
                claim_value=payload,
            )
        )
        before = ledger.query_as_of("pv7-3d", "shadow-marks", 10, 5)
        frozen = ledger.query_as_of("pv7-3d", "shadow-marks", 10, 10)
    if before.status != "ABSENT" or frozen.status != "RESOLVED" or frozen.winner_claim_value != payload:
        raise RuntimeError("SELECT mark freeze was visible before its transaction time")
    store = StratifiedEvidenceStore()
    store.apply_mutation(
        [
            StratifiedEvent(
                event_id="pm6-shadow-select-freeze",
                entity="shadow-marks",
                slot="detector",
                claim_value=payload,
                text="select freeze",
                valid_time=10,
                tx_time=10,
                tenant="pv7-3d",
                domain="probe",
                authority_tier=1,
            )
        ]
    )
    trace = export_canonical_ek5_audit_trace(store)
    identity = TRA1StateIdentityContract(
        base_weights_sha256=sha256_digest(payload.encode("utf-8")),
        adapter_weights_sha256=sha256_digest(b"pv7-3d-shadow-no-adapter"),
        config_sha256=hashlib.sha256(b"pv7-w18-3d-shadow-config-v1").hexdigest(),
        tokenizer_sha256=hashlib.sha256(b"pv7-w18-3d-scratch-tokenizer-v1").hexdigest(),
        kv_or_state_dtype="float64",
        rope_freq_base=10000.0,
        rope_scaling_hash=hashlib.sha256(b"pv7-3d-rope-none").hexdigest(),
    )
    return {"as_of_digest": frozen.digest(), "ek5_trace_digest": trace.digest(), "tra1_digest": identity.digest(), "claim_value": payload}


def _probe(marks: Sequence[str], flags: List[bool]) -> Probe:
    frozen = tuple(marks)

    def probe(envelope: ShadowTapEnvelope) -> bytes:
        text = bytes(envelope.stream).decode("utf-8")
        flags.append(_marks_present(text, frozen))
        return b"TAMPER"

    return probe


def _identity_probes(sample: str) -> Dict[str, Any]:
    alone = pm6_release_bytes(sample, probe=None)

    def mutate(envelope: ShadowTapEnvelope) -> bytes:
        envelope.hidden += 100.0
        envelope.stream.extend(b"TAMPER")
        envelope.decision["action"] = "abort"
        return b"TAMPER"

    def explode(envelope: ShadowTapEnvelope) -> None:
        envelope.hidden += 1.0
        raise RuntimeError("shadow probe raised")

    mutated = pm6_release_bytes(sample, probe=mutate)
    thrown = pm6_release_bytes(sample, probe=explode)
    envelope = ShadowTapEnvelope.capture(_hidden_from_text(sample), sample.encode("utf-8"), {"action": "release"})
    returned = shadow_gate_return(envelope, mutate)
    return {
        "release_bytes_sha256": hashlib.sha256(alone).hexdigest(),
        "release_byte_length": len(alone),
        "mutation_identical": alone == mutated and len(alone) > 0,
        "throw_identical": alone == thrown,
        "shadow_return_is_none": returned is None,
    }


def _sensitivity_and_lead_time() -> Dict[str, Any]:
    subjects = _SUBJECTS * (SENSITIVITY_COUNT // len(_SUBJECTS) + 1)
    hits_1 = 0
    hits_4 = 0
    leads: List[int] = []
    for index in range(SENSITIVITY_COUNT):
        subject = subjects[index]
        text = f"Describe {subject} number {index} then {SINGLETON_MARK} and stop."
        hits_1 += int(_marks_present(text, (SINGLETON_MARK,)))
        sparse_text = f"Describe {subject} number {index} then {SPARSE_MARKS[index % 4]} and stop."
        hits_4 += int(_marks_present(sparse_text, SPARSE_MARKS))
    for index in range(R3_COUNT):
        subject = _DOMAINS[index % len(_DOMAINS)]
        source = f"State the {subject} status {SINGLETON_MARK} now."
        paraphrase = f"Give the present condition of the {subject} and then include {SINGLETON_MARK} exactly."
        source_at = source.split().index(SINGLETON_MARK)
        paraphrase_at = paraphrase.split().index(SINGLETON_MARK)
        leads.append(paraphrase_at - source_at)
    ordered = sorted(leads)
    median = ordered[len(ordered) // 2]
    return {
        "sensitivity_n": SENSITIVITY_COUNT,
        "singleton_hits": hits_1,
        "sparse_hits": hits_4,
        "counted_in_false_alert_denominator": False,
        "r3_lead_times": leads,
        "r3_lead_min": ordered[0],
        "r3_lead_median": median,
        "r3_lead_max": ordered[-1],
        "r3_tier": "T3",
        "r3_confirmatory": False,
    }


def _splits_disjoint(family_ids: Mapping[str, Sequence[str]]) -> bool:
    seen: set[str] = set()
    for name, rows in family_ids.items():
        del name
        if len(rows) != len(set(rows)):
            return False
        if seen.intersection(rows):
            return False
        seen.update(rows)
    return set(family_ids) == {"dev", "select", "report"} and len(family_ids["report"]) >= 300


def run_shadow_drycheck() -> Dict[str, Any]:
    splits = partition_splits()
    families = [row for rows in splits.values() for row in rows]
    burn_sha = admit_scratch_prompts(families)
    dev_alerts_1 = sum(_marks_present(row["prompt"], (SINGLETON_MARK,)) for row in splits["dev"])
    dev_alerts_4 = sum(_marks_present(row["prompt"], SPARSE_MARKS) for row in splits["dev"])
    if dev_alerts_1 or dev_alerts_4:
        raise RuntimeError("DEV calibration saw a detector mark in a benign family")
    freeze = _freeze_marks()
    flags_1: List[bool] = []
    flags_4: List[bool] = []
    identical = 0
    incumbent_alerts = 0
    for row in splits["report"]:
        text = row["prompt"]
        alone = pm6_release_bytes(text, probe=None)
        with_1 = pm6_release_bytes(text, probe=_probe((SINGLETON_MARK,), flags_1))
        with_4 = pm6_release_bytes(text, probe=_probe(SPARSE_MARKS, flags_4))
        if alone == with_1 == with_4 and len(alone) > 0:
            identical += 1
        if len(alone) == 0:
            incumbent_alerts += 1
    alerts_1 = int(sum(flags_1))
    alerts_4 = int(sum(flags_4))
    record_1 = _false_alert_record("1unit", alerts_1, REPORT_COUNT)
    record_4 = _false_alert_record("4unit", alerts_4, REPORT_COUNT)
    adjusted = holm_bonferroni_adjust(
        {
            "1unit": record_1["raw_p_value_rate_at_least_limit"],
            "4unit": record_4["raw_p_value_rate_at_least_limit"],
        }
    )
    record_1["holm_adjusted_p_value"] = adjusted["1unit"]
    record_4["holm_adjusted_p_value"] = adjusted["4unit"]
    telemetry = _sensitivity_and_lead_time()
    identity = _identity_probes("alpha beta gamma delta epsilon zeta")
    return {
        "bundle_id": BUNDLE_ID,
        "item": ITEM,
        "report_family_count": REPORT_COUNT,
        "family_ids": {name: [row["family_id"] for row in rows] for name, rows in splits.items()},
        "burn_ledger_sha256": burn_sha,
        "select_freeze": {key: freeze[key] for key in ("as_of_digest", "ek5_trace_digest", "tra1_digest")},
        "dev_false_alerts": {"1unit": dev_alerts_1, "4unit": dev_alerts_4},
        "incumbent_pm6_alone": {"families": REPORT_COUNT, "empty_release_count": incumbent_alerts},
        "incumbent_pm6_plus_shadow": {"1unit": record_1, "4unit": record_4},
        "byte_identical_families": identical,
        "identity_probe": identity,
        "telemetry": telemetry,
        "live_pm6_gateway_executed": False,
    }


def _gates(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    one = report["incumbent_pm6_plus_shadow"]["1unit"]
    four = report["incumbent_pm6_plus_shadow"]["4unit"]
    identity = report["identity_probe"]
    telemetry = report["telemetry"]
    checks = [
        ("shadow_returns_none", identity["shadow_return_is_none"], "gate received None"),
        ("mutation_bytes_identical", identity["mutation_identical"], identity["release_bytes_sha256"]),
        ("throw_bytes_identical", identity["throw_identical"], "exception stayed inside the tap"),
        ("families_at_least_320", report["report_family_count"] >= 320, str(report["report_family_count"])),
        ("splits_disjoint", _splits_disjoint(report["family_ids"]), "dev select report"),
        ("singleton_holm_cp_upper", one["holm_adjusted_cp95_upper"] < FALSE_ALERT_LIMIT, str(one["holm_adjusted_cp95_upper"])),
        ("sparse_holm_cp_upper", four["holm_adjusted_cp95_upper"] < FALSE_ALERT_LIMIT, str(four["holm_adjusted_cp95_upper"])),
        ("holm_p_values", one["holm_adjusted_p_value"] < FALSE_ALERT_ALPHA and four["holm_adjusted_p_value"] < FALSE_ALERT_ALPHA, str(one["holm_adjusted_p_value"])),
        ("byte_identity_all_families", report["byte_identical_families"] == report["report_family_count"], str(report["byte_identical_families"])),
        ("incumbent_released", report["incumbent_pm6_alone"]["empty_release_count"] == 0, "benign scratch text released"),
        ("sensitivity_separate", telemetry["counted_in_false_alert_denominator"] is False and telemetry["singleton_hits"] == telemetry["sensitivity_n"] and telemetry["sparse_hits"] == telemetry["sensitivity_n"], "sensitivity held out"),
        ("r3_observational_only", telemetry["r3_confirmatory"] is False and telemetry["r3_tier"] == "T3" and telemetry["r3_lead_min"] > 0, f"median={telemetry['r3_lead_median']}"),
        ("cuda_isolated", bool(report["cuda_isolation"]["passed"]), "CPU lease"),
        ("live_gateway_not_claimed", report["live_pm6_gateway_executed"] is False, "drycheck only"),
        ("burn_ledger_admitted", bool(report["burn_ledger_sha256"]), report["burn_ledger_sha256"]),
    ]
    return [{"name": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks]


def _fragment_digest() -> str:
    registry = REPO_ROOT / "docs" / "research" / "registry.d" / "53-t3d-pm6-shadow.json"
    verifier = REPO_ROOT / "scripts" / "verifiers.d" / "53_verifier.py"
    sources = {
        "docs/research/registry.d/53-t3d-pm6-shadow.json": registry.read_bytes(),
        "scripts/verifiers.d/53_verifier.py": verifier.read_bytes(),
    }
    compiled = compile_fragments(REPO_ROOT, claim_registry_sha256(), sources)
    return str(compiled["compiled_sha256"])


def _protected_manifest(path: Path) -> bool:
    return any(any(token in part.casefold() for token in _PROTECTED_PARTS) for part in path.parts)


def execute_live_shadow(manifest: Optional[str]) -> Dict[str, Any]:
    """Refuse a protected path before opening it. A missing manifest does not invent a live result."""
    if not manifest:
        return {
            "executed": False,
            "report_emitted": False,
            "reason": "manifest-required",
            "live_pm6_gateway_executed": False,
            "claim_level": "not-executed",
        }
    path = Path(manifest)
    if _protected_manifest(path):
        return {
            "executed": False,
            "report_emitted": False,
            "reason": "protected-path-refused",
            "opened": False,
            "live_pm6_gateway_executed": False,
            "claim_level": "not-executed",
        }
    if not path.is_file():
        return {
            "executed": False,
            "report_emitted": False,
            "reason": "manifest-absent",
            "live_pm6_gateway_executed": False,
            "claim_level": "not-executed",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, list) or not prompts or any(not isinstance(item, str) or not item for item in prompts):
        return {
            "executed": False,
            "report_emitted": False,
            "reason": "manifest-prompts-missing",
            "live_pm6_gateway_executed": False,
            "claim_level": "not-executed",
        }
    flags: List[bool] = []
    identical = 0
    for text in prompts:
        alone = pm6_release_bytes(text, probe=None)
        shadowed = pm6_release_bytes(text, probe=_probe((SINGLETON_MARK,), flags))
        identical += int(alone == shadowed)
    return {
        "executed": True,
        "report_emitted": False,
        "reason": "manifest-shadow-observed",
        "families": len(prompts),
        "byte_identical_families": identical,
        "shadow_flags": int(sum(flags)),
        "live_pm6_gateway_executed": False,
        "claim_level": "observational-manifest",
    }


def execute(mode: str, pm6_manifest: Optional[str] = None) -> Dict[str, Any]:
    if mode not in {"local-drycheck", "run-live-shadow"}:
        raise ValueError("mode must be local-drycheck or run-live-shadow")
    isolation = cpu_isolation_receipt()
    if mode == "run-live-shadow":
        report = execute_live_shadow(pm6_manifest)
        report.update({"bundle_id": BUNDLE_ID, "item": ITEM, "mode": mode, "cuda_isolation": isolation})
        report["gates"] = [
            {
                "name": "live_shadow_does_not_invent_gateway_result",
                "passed": report["live_pm6_gateway_executed"] is False,
                "detail": report["reason"],
            }
        ]
        report["passed"] = all(gate["passed"] for gate in report["gates"])
        report["scoreboard_sha256"] = _json_sha256({key: value for key, value in report.items() if key != "scoreboard_sha256"})
        return report

    dry = run_shadow_drycheck()
    report = {
        **dry,
        "mode": mode,
        "executed": True,
        "cuda_isolation": isolation,
        "claim_level": "local-drycheck",
        "fragment_compiled_sha256": _fragment_digest(),
        "spine_digest": claim_registry_sha256(),
    }
    report["gates"] = _gates(report)
    report["passed"] = all(gate["passed"] for gate in report["gates"])
    report["scoreboard_sha256"] = _json_sha256({key: value for key, value in report.items() if key != "scoreboard_sha256"})
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("local-drycheck", "run-live-shadow"))
    parser.add_argument("--pm6-manifest", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = execute(args.mode, args.pm6_manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
