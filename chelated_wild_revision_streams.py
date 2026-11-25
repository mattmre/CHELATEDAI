"""Deterministic ingestors for wild advisory and SEC revision streams.

No authored prompts and no model write tokens. Parsers accept OSV, GHSA, NVD,
CSAF VEX, and SEC EDGAR submission documents. Adjudication is a separate
labeled envelope. Synthetic schema fixtures are never marked wild.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from ek5_v3_stratified import classify_high_similarity_pair

STRATA = (
    "unlinked_cross_mirror_echo",
    "equal_authority_vex_ghsa",
    "sec_item_402_nonreliance",
    "routine_amendment_control",
)
CUT_NAMES = (
    "t_initial",
    "t_mirror_echo_or_conflict",
    "t_withdrawal_or_402",
    "t_reinstatement_or_restated_10ka",
)
WILD_FAMILY_MINIMUM = 240
WILD_PER_STRATUM = 60
COHORT_DRYCHECK = "deterministic_schema_drycheck"
COHORT_WILD = "frozen_wild_snapshots"
COHORT_DESCRIPTIVE = "descriptive_partial_snapshot"
STATUS_DRYCHECK = "DRYCHECK"
ASSERTING = frozenset({"ASSERT", "CONFLICT"})
PROTECTED_NAME_PARTS = ("sealed_key", "blind-followup", "pm4_human_audit_100_blind_sheet.csv")

ORIGIN_TEXT = (
    "Vendor advisory GHSADEMO1001 reports the widget package is affected before the patched release."
)
ECHO_TEXT = "Briefing 1: " + ORIGIN_TEXT
ECHO_TEXT_2 = "Memory recap 2: " + ORIGIN_TEXT
REINSTATE_TEXT = (
    "Reinstated vendor advisory GHSADEMO1001 reports the widget package is affected before release 9.9.9."
)
GHSA_TEXT = "GitHub advisory GHSADEMO2002 says the widget package is affected at release 2.0.0."
GHSA_RESTORED_TEXT = "GitHub advisory GHSADEMO2002 says the widget package is affected again at release 2.0.1."
VEX_TEXT = "CSAF VEX record GHSADEMO2002 says the widget package is not affected at release 2.0.0."
TENK_TEXT = "Northwind annual report states the fiscal results remain reliable for the filed year."
PRESS_TEXT = "Briefing 3: " + TENK_TEXT
ITEM_402_TEXT = "Form 8-K item 4.02 states prior Northwind fiscal results should no longer be relied upon."
RESTATED_TEXT = "Restated Northwind report states corrected revenue of 364 usd is the current reliable figure."
AMEND_TEXT = (
    "Form 10-K amendment corrects exhibit 21 only and leaves the filed Northwind fiscal results unchanged."
)
AMEND_FOLLOWUP_TEXT = (
    "Form 8-K amendment updates officer code 88 only and leaves the filed Northwind fiscal results unchanged."
)


class StreamParseError(ValueError):
    """A snapshot document does not match a supported advisory or filing schema."""


@dataclass(frozen=True)
class NormalizedAdvisory:
    source: str
    advisory_id: str
    status: str
    summary: str
    product: str
    version_range: str
    published: str
    modified: str
    withdrawn: bool
    aliases: Tuple[str, ...]
    form: str
    items: Tuple[str, ...]
    amendment: bool
    sec_class: str

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "advisory_id": self.advisory_id,
            "status": self.status,
            "summary": self.summary,
            "product": self.product,
            "version_range": self.version_range,
            "published": self.published,
            "modified": self.modified,
            "withdrawn": self.withdrawn,
            "aliases": list(self.aliases),
            "form": self.form,
            "items": list(self.items),
            "amendment": self.amendment,
            "sec_class": self.sec_class,
        }


@dataclass(frozen=True)
class StreamRecord:
    record_id: str
    tenant_id: str
    entity_key: str
    claim_value: str
    authority_rank: int
    valid_from: int
    valid_to: Optional[int]
    tx_time: int
    op_kind: str
    text: str
    source: str
    target_id: str = ""
    echo_of: str = ""
    explicit_parent_ids: Tuple[str, ...] = ()
    supersedes_ids: Tuple[str, ...] = ()
    product: str = ""
    version: str = ""
    form: str = ""
    items: Tuple[str, ...] = ()
    slot: str = "status"

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "tenant_id": self.tenant_id,
            "entity_key": self.entity_key,
            "claim_value": self.claim_value,
            "authority_rank": self.authority_rank,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "tx_time": self.tx_time,
            "op_kind": self.op_kind,
            "text": self.text,
            "source": self.source,
            "target_id": self.target_id,
            "echo_of": self.echo_of,
            "explicit_parent_ids": list(self.explicit_parent_ids),
            "supersedes_ids": list(self.supersedes_ids),
            "product": self.product,
            "version": self.version,
            "form": self.form,
            "items": list(self.items),
            "slot": self.slot,
        }


@dataclass(frozen=True)
class CutSpec:
    name: str
    valid_time: int
    tx_time: int

    def to_dict(self) -> dict:
        return {"name": self.name, "valid_time": self.valid_time, "tx_time": self.tx_time}


@dataclass(frozen=True)
class CutOracle:
    name: str
    valid_time: int
    tx_time: int
    disposition: str
    independent_lineage_ids: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "valid_time": self.valid_time,
            "tx_time": self.tx_time,
            "disposition": self.disposition,
            "independent_lineage_ids": list(self.independent_lineage_ids),
        }


@dataclass(frozen=True)
class AdjudicatedFamily:
    family_id: str
    stratum: str
    tenant_id: str
    entity_key: str
    records: Tuple[StreamRecord, ...]
    cuts: Tuple[CutSpec, ...]
    adjudicated: bool = True
    synthetic: bool = False

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "stratum": self.stratum,
            "tenant_id": self.tenant_id,
            "entity_key": self.entity_key,
            "records": [record.to_dict() for record in self.records],
            "cuts": [cut.to_dict() for cut in self.cuts],
            "adjudicated": self.adjudicated,
            "synthetic": self.synthetic,
            "cohort_source": COHORT_DRYCHECK if self.synthetic else COHORT_WILD,
        }


@dataclass(frozen=True)
class EchoPair:
    left_text: str
    right_text: str
    left_tenant: str
    right_tenant: str
    expect_collapse: bool

    @property
    def cross_tenant(self) -> bool:
        return self.left_tenant != self.right_tenant


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    return {}


def classify_sec_filing(form: str, items: Iterable[object]) -> str:
    """Separate a true 8-K Item 4.02 from routine 10-K/A and 8-K/A amendments.

    An amended form never opens a new non-reliance withdrawal by itself.
    Item 4.02 on a plain 8-K does.
    """
    normalized = _text(form).upper().replace(" ", "")
    amendment = normalized.endswith("/A")
    base = normalized[:-2] if amendment else normalized
    tokens = set()
    for item in items:
        for part in _text(item).replace(";", ",").split(","):
            token = part.strip().lower().replace("item", "").strip()
            if token:
                tokens.add(token)
    has_402 = "4.02" in tokens or "402" in tokens
    if base == "8-K" and has_402 and not amendment:
        return "item_402_nonreliance"
    if amendment and has_402:
        return "item_402_amendment_does_not_open_withdrawal"
    if amendment:
        return "routine_amendment"
    if base == "8-K":
        return "ordinary_8k"
    if base.startswith("10-K"):
        return "periodic_report"
    return "other_filing"


def _advisory(
    *,
    source: str,
    advisory_id: str,
    status: str,
    summary: str = "",
    product: str = "",
    version_range: str = "",
    published: str = "",
    modified: str = "",
    withdrawn: bool = False,
    aliases: Iterable[str] = (),
    form: str = "",
    items: Iterable[str] = (),
    amendment: bool = False,
    sec_class: str = "",
) -> NormalizedAdvisory:
    if not advisory_id:
        raise StreamParseError(f"{source} document is missing an identifier")
    return NormalizedAdvisory(
        source=source,
        advisory_id=advisory_id,
        status="withdrawn" if withdrawn else status,
        summary=summary,
        product=product,
        version_range=version_range,
        published=published,
        modified=modified,
        withdrawn=withdrawn,
        aliases=tuple(alias for alias in (_text(item) for item in aliases) if alias),
        form=form,
        items=tuple(item for item in (_text(item) for item in items) if item),
        amendment=amendment,
        sec_class=sec_class,
    )


def _osv_package(affected: object) -> Tuple[str, str]:
    product = ""
    version_range = ""
    for entry in _as_list(affected):
        if not isinstance(entry, dict):
            continue
        package = entry.get("package")
        if isinstance(package, dict) and not product:
            product = _text(package.get("name") or package.get("ecosystem"))
        if not version_range:
            versions = entry.get("versions")
            if isinstance(versions, list) and versions:
                version_range = _text(versions[0])
            for rng in _as_list(entry.get("ranges")):
                if not isinstance(rng, dict):
                    continue
                events = [item for item in _as_list(rng.get("events")) if isinstance(item, dict)]
                introduced = next((_text(item.get("introduced")) for item in events if item.get("introduced")), "")
                fixed = next((_text(item.get("fixed")) for item in events if item.get("fixed")), "")
                if introduced or fixed:
                    version_range = f"[{introduced or '0'},{fixed or 'inf'})"
                    break
        if product and version_range:
            break
    return product, version_range


def parse_osv_advisory(document: Mapping[str, object]) -> NormalizedAdvisory:
    """Parse one OSV schema document. ``withdrawn`` forces status withdrawn."""
    if not isinstance(document, Mapping):
        raise StreamParseError("OSV advisory must be an object")
    product, version_range = _osv_package(document.get("affected"))
    aliases = _as_list(document.get("aliases")) + _as_list(document.get("related"))
    withdrawn = bool(_text(document.get("withdrawn")))
    status = "withdrawn" if withdrawn else "affected"
    database_specific = _first_mapping(document.get("database_specific"))
    if not withdrawn and _text(database_specific.get("status")).lower() in {"not_affected", "unaffected"}:
        status = "not_affected"
    return _advisory(
        source="osv",
        advisory_id=_text(document.get("id")),
        status=status,
        summary=_text(document.get("summary") or document.get("details")),
        product=product,
        version_range=version_range,
        published=_text(document.get("published")),
        modified=_text(document.get("modified")),
        withdrawn=withdrawn,
        aliases=[_text(item) for item in aliases],
    )


def parse_osv_seed_cache(payload: Mapping[str, object]) -> Tuple[NormalizedAdvisory, ...]:
    """Parse the simplified ``cache/osv_cve_seeds.json`` object or a list of OSV docs."""
    if not isinstance(payload, Mapping):
        raise StreamParseError("OSV seed cache must be an object")
    advisories = []
    values = list(payload.values())
    simplified = values and all(isinstance(item, Mapping) and ("osv_summary" in item or "cve_id" in item) for item in values)
    if simplified and "id" not in payload and "affected" not in payload:
        for key, item in payload.items():
            if not isinstance(item, Mapping):
                continue
            advisories.append(
                _advisory(
                    source="osv_seed_cache",
                    advisory_id=_text(item.get("cve_id") or key),
                    status="affected",
                    summary=_text(item.get("osv_summary")),
                    product=_text(item.get("osv_package")),
                    version_range=_text(item.get("osv_fixed")),
                )
            )
        return tuple(advisories)
    if "id" in payload or "affected" in payload:
        return (parse_osv_advisory(payload),)
    for item in values:
        if isinstance(item, Mapping) and ("id" in item or "affected" in item):
            advisories.append(parse_osv_advisory(item))
    return tuple(advisories)


def parse_ghsa_advisory(document: Mapping[str, object]) -> NormalizedAdvisory:
    """Parse a GitHub Security Advisory object."""
    if not isinstance(document, Mapping):
        raise StreamParseError("GHSA advisory must be an object")
    product = ""
    version_range = ""
    for vuln in _as_list(document.get("vulnerabilities")):
        if not isinstance(vuln, Mapping):
            continue
        package = vuln.get("package")
        if isinstance(package, Mapping) and not product:
            product = _text(package.get("name"))
        version_range = version_range or _text(vuln.get("vulnerableVersionRange") or vuln.get("patchedVersion"))
    withdrawn = bool(_text(document.get("withdrawnAt") or document.get("withdrawn_at")))
    severity = _text(document.get("severity")).lower()
    status = "withdrawn" if withdrawn else "affected"
    if not withdrawn and severity in {"not_affected", "unaffected"}:
        status = "not_affected"
    aliases = [_text(document.get("cveId") or document.get("cve_id"))]
    return _advisory(
        source="ghsa",
        advisory_id=_text(document.get("ghsaId") or document.get("ghsa_id") or document.get("id")),
        status=status,
        summary=_text(document.get("summary") or document.get("description")),
        product=product,
        version_range=version_range,
        published=_text(document.get("publishedAt") or document.get("published_at")),
        modified=_text(document.get("updatedAt") or document.get("updated_at")),
        withdrawn=withdrawn,
        aliases=aliases,
    )


def _nvd_status(cve: Mapping[str, object]) -> Tuple[str, bool]:
    raw = _text(cve.get("vulnStatus") or cve.get("vuln_status")).lower()
    if raw in {"rejected", "withdrawn"}:
        return "withdrawn", True
    if raw in {"not_affected", "unaffected"}:
        return "not_affected", False
    return "affected", False


def parse_nvd_cve(document: Mapping[str, object]) -> Tuple[NormalizedAdvisory, ...]:
    """Parse an NVD 2.0 CVE object or a ``vulnerabilities`` wrapper."""
    if not isinstance(document, Mapping):
        raise StreamParseError("NVD document must be an object")
    cves = []
    wrapped = document.get("vulnerabilities")
    if isinstance(wrapped, list):
        for item in wrapped:
            if isinstance(item, Mapping):
                cve = item.get("cve") if isinstance(item.get("cve"), Mapping) else item
                if isinstance(cve, Mapping):
                    cves.append(cve)
    elif isinstance(document.get("cve"), Mapping):
        cves.append(document["cve"])
    elif "id" in document and "vulnStatus" in document:
        cves.append(document)
    if not cves:
        raise StreamParseError("NVD document did not contain a CVE")
    parsed = []
    for cve in cves:
        status, withdrawn = _nvd_status(cve)
        summary = ""
        for description in _as_list(cve.get("descriptions")):
            if isinstance(description, Mapping) and _text(description.get("lang")).lower() in {"", "en"}:
                summary = _text(description.get("value"))
                break
        product = ""
        for config in _as_list(cve.get("configurations")):
            if not isinstance(config, Mapping):
                continue
            for node in _as_list(config.get("nodes")):
                if not isinstance(node, Mapping):
                    continue
                for match in _as_list(node.get("cpeMatch")):
                    if isinstance(match, Mapping) and match.get("criteria"):
                        product = _text(match.get("criteria"))
                        break
                if product:
                    break
        parsed.append(
            _advisory(
                source="nvd",
                advisory_id=_text(cve.get("id")),
                status=status,
                summary=summary,
                product=product,
                published=_text(cve.get("published")),
                modified=_text(cve.get("lastModified")),
                withdrawn=withdrawn,
            )
        )
    return tuple(parsed)


def _csaf_statuses(vuln: Mapping[str, object]) -> list:
    product_status = vuln.get("product_status")
    found = []
    if isinstance(product_status, Mapping):
        if product_status.get("known_affected") or product_status.get("affected"):
            found.append(("affected", product_status.get("known_affected") or product_status.get("affected")))
        if product_status.get("known_not_affected") or product_status.get("not_affected"):
            found.append(("not_affected", product_status.get("known_not_affected") or product_status.get("not_affected")))
    if not found:
        flags = [item for item in _as_list(vuln.get("flags")) if isinstance(item, Mapping)]
        for flag in flags:
            label = _text(flag.get("label")).lower()
            if label in {"vulnerable_code_not_present", "vulnerable_code_not_in_execute_path", "component_not_present"}:
                found.append(("not_affected", flag.get("product_ids")))
        if not found:
            found.append(("affected", ()))
    return found


def parse_csaf_vex(document: Mapping[str, object]) -> Tuple[NormalizedAdvisory, ...]:
    """Parse a CSAF VEX document into one advisory per product-status claim."""
    if not isinstance(document, Mapping):
        raise StreamParseError("CSAF document must be an object")
    doc = document.get("document") if isinstance(document.get("document"), Mapping) else {}
    tracking = doc.get("tracking") if isinstance(doc.get("tracking"), Mapping) else {}
    tracking_id = _text(tracking.get("id") or doc.get("title"))
    published = _text(tracking.get("initial_release_date"))
    modified = _text(tracking.get("current_release_date"))
    vulnerabilities = document.get("vulnerabilities")
    if not isinstance(vulnerabilities, list) or not vulnerabilities:
        raise StreamParseError("CSAF VEX document requires vulnerabilities")
    parsed = []
    for vuln in vulnerabilities:
        if not isinstance(vuln, Mapping):
            continue
        vuln_id = _text(vuln.get("cve") or vuln.get("id") or tracking_id)
        notes = ""
        for note in _as_list(vuln.get("notes")):
            if isinstance(note, Mapping):
                notes = _text(note.get("text"))
                if notes:
                    break
        for status, products in _csaf_statuses(vuln):
            product_ids = [_text(item) for item in _as_list(products)]
            product = product_ids[0] if product_ids else ""
            suffix = "" if status == "affected" else f":{status}"
            parsed.append(
                _advisory(
                    source="csaf_vex",
                    advisory_id=f"{vuln_id}{suffix}" if suffix else vuln_id,
                    status=status,
                    summary=notes or _text(doc.get("title")),
                    product=product,
                    published=published,
                    modified=modified,
                    aliases=[vuln_id, tracking_id],
                )
            )
    if not parsed:
        raise StreamParseError("CSAF VEX document did not contain a vulnerability claim")
    return tuple(parsed)


def _sec_record(form: str, accession: str, filed: str, accepted: str, items: Iterable[object], summary: str) -> NormalizedAdvisory:
    item_list = []
    for item in items:
        for part in _text(item).replace(";", ",").split(","):
            token = part.strip()
            if token:
                item_list.append(token)
    sec_class = classify_sec_filing(form, item_list)
    amendment = _text(form).upper().replace(" ", "").endswith("/A")
    if sec_class == "item_402_nonreliance":
        status = "non_reliance"
    elif sec_class == "routine_amendment":
        status = "reliance_unchanged"
    elif sec_class == "item_402_amendment_does_not_open_withdrawal":
        status = "reliance_unchanged"
    else:
        status = "filed"
    return _advisory(
        source="sec_edgar",
        advisory_id=accession or f"{form}:{filed}",
        status=status,
        summary=summary,
        published=filed,
        modified=accepted or filed,
        form=_text(form).upper(),
        items=item_list,
        amendment=amendment,
        sec_class=sec_class,
    )


def parse_sec_submission(document: Mapping[str, object]) -> Tuple[NormalizedAdvisory, ...]:
    """Parse one filing or an EDGAR ``filings.recent`` columnar submission.

    True Item 4.02 non-reliance is only a non-amended 8-K whose items include
    4.02. Routine 10-K/A and 8-K/A filings stay amendment controls.
    """
    if not isinstance(document, Mapping):
        raise StreamParseError("SEC submission must be an object")
    filings = document.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    if isinstance(recent, Mapping) and isinstance(recent.get("form"), list):
        forms = recent.get("form") or []
        accessions = recent.get("accessionNumber") or []
        dates = recent.get("filingDate") or []
        accepted = recent.get("acceptanceDateTime") or []
        items = recent.get("items") or []
        docs = recent.get("primaryDocument") or []
        parsed = []
        for index, form in enumerate(forms):
            parsed.append(
                _sec_record(
                    _text(form),
                    _text(accessions[index] if index < len(accessions) else ""),
                    _text(dates[index] if index < len(dates) else ""),
                    _text(accepted[index] if index < len(accepted) else ""),
                    [_text(items[index])] if index < len(items) else [],
                    _text(docs[index] if index < len(docs) else form),
                )
            )
        if not parsed:
            raise StreamParseError("SEC submissions recent list was empty")
        return tuple(parsed)
    form = _text(document.get("form") or document.get("formType"))
    if not form:
        raise StreamParseError("SEC filing is missing form")
    return (
        _sec_record(
            form,
            _text(document.get("accessionNumber") or document.get("accession")),
            _text(document.get("filingDate") or document.get("filed")),
            _text(document.get("acceptanceDateTime")),
            _as_list(document.get("items") if document.get("items") is not None else document.get("item")),
            _text(document.get("primaryDocument") or document.get("summary") or form),
        ),
    )


def parse_snapshot_document(document: object) -> Tuple[NormalizedAdvisory, ...]:
    """Dispatch one JSON object to the matching advisory or filing parser."""
    if isinstance(document, list):
        parsed = []
        for item in document:
            parsed.extend(parse_snapshot_document(item))
        return tuple(parsed)
    if not isinstance(document, Mapping):
        raise StreamParseError("snapshot document must be a JSON object or list")
    if isinstance(document.get("filings"), Mapping) or document.get("form") or document.get("formType"):
        return parse_sec_submission(document)
    if isinstance(document.get("document"), Mapping) and isinstance(document.get("vulnerabilities"), list):
        return parse_csaf_vex(document)
    if document.get("ghsaId") or document.get("ghsa_id"):
        return (parse_ghsa_advisory(document),)
    if isinstance(document.get("vulnerabilities"), list) or document.get("vulnStatus") or (
        isinstance(document.get("cve"), Mapping) and "vulnStatus" in document.get("cve")
    ):
        return parse_nvd_cve(document)
    if "osv_summary" in document or (
        document and all(isinstance(item, Mapping) and "osv_summary" in item for item in document.values())
    ):
        return parse_osv_seed_cache(document)
    if document.get("id") or "affected" in document:
        return (parse_osv_advisory(document),)
    raise StreamParseError("unrecognized advisory or filing schema")


def load_json_snapshot(path: Path) -> object:
    """Read one UTF-8 JSON snapshot. Refuses protected filenames."""
    if any(part.casefold() in {name.casefold() for name in PROTECTED_NAME_PARTS} or any(
        forbidden in part.casefold() for forbidden in ("sealed_key", "blind-followup")
    ) for part in path.parts):
        raise StreamParseError(f"refusing protected snapshot path: {path.name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_osv_seed_cache(path: Path) -> Tuple[NormalizedAdvisory, ...]:
    """Load a seed cache when it exists. A missing cache yields no advisories."""
    if not path.is_file():
        return ()
    payload = load_json_snapshot(path)
    if not isinstance(payload, Mapping):
        raise StreamParseError("OSV seed cache must be a JSON object")
    return parse_osv_seed_cache(payload)


def find_equal_authority_conflicts(advisories: Sequence[NormalizedAdvisory]) -> Tuple[dict, ...]:
    """Group incompatible affected and not_affected claims on one product and version.

    Equal-authority product conflicts are Disputed. The parser does not pick a winner.
    """
    groups = defaultdict(list)
    for advisory in advisories:
        if advisory.status not in {"affected", "not_affected"}:
            continue
        key = (advisory.product.casefold(), advisory.version_range.casefold())
        if not key[0]:
            continue
        groups[key].append(advisory)
    conflicts = []
    for (product, version), rows in sorted(groups.items()):
        statuses = {row.status for row in rows}
        if statuses == {"affected", "not_affected"} and len({row.source for row in rows}) >= 1:
            conflicts.append(
                {
                    "product": product,
                    "version": version,
                    "disposition": "disputed",
                    "advisory_ids": sorted(row.advisory_id for row in rows),
                    "sources": sorted({row.source for row in rows}),
                }
            )
    return tuple(conflicts)


def _record_from_mapping(raw: Mapping[str, object], family_tenant: str, family_entity: str) -> StreamRecord:
    op_kind = _text(raw.get("op_kind") or "ASSERT").upper()
    if op_kind not in {"ASSERT", "CONFLICT", "WITHDRAW", "REINSTATE"}:
        raise StreamParseError(f"unsupported op_kind {op_kind}")
    valid_to = raw.get("valid_to")
    return StreamRecord(
        record_id=_text(raw.get("record_id")),
        tenant_id=_text(raw.get("tenant_id") or family_tenant),
        entity_key=_text(raw.get("entity_key") or family_entity),
        claim_value=_text(raw.get("claim_value")),
        authority_rank=int(raw.get("authority_rank") or 0),
        valid_from=int(raw.get("valid_from") or 0),
        valid_to=None if valid_to in (None, "") else int(valid_to),
        tx_time=int(raw.get("tx_time") or 0),
        op_kind=op_kind,
        text=_text(raw.get("text")),
        source=_text(raw.get("source") or "snapshot"),
        target_id=_text(raw.get("target_id")),
        echo_of=_text(raw.get("echo_of")),
        explicit_parent_ids=tuple(_text(item) for item in _as_list(raw.get("explicit_parent_ids")) if _text(item)),
        supersedes_ids=tuple(_text(item) for item in _as_list(raw.get("supersedes_ids")) if _text(item)),
        product=_text(raw.get("product")),
        version=_text(raw.get("version")),
        form=_text(raw.get("form")),
        items=tuple(_text(item) for item in _as_list(raw.get("items")) if _text(item)),
        slot=_text(raw.get("slot") or "status"),
    )


def parse_family_envelope(document: Mapping[str, object], *, synthetic: bool = False) -> Optional[AdjudicatedFamily]:
    """Parse an adjudicated family envelope. Incomplete envelopes return None."""
    if not isinstance(document, Mapping) or "family_id" not in document or "stratum" not in document:
        return None
    stratum = _text(document.get("stratum"))
    if stratum not in STRATA:
        raise StreamParseError(f"unknown stratum {stratum}")
    if document.get("adjudicated") is not True:
        return None
    cuts_raw = document.get("cuts")
    records_raw = document.get("records")
    if not isinstance(cuts_raw, list) or not isinstance(records_raw, list):
        return None
    by_name = {}
    for cut in cuts_raw:
        if not isinstance(cut, Mapping):
            continue
        name = _text(cut.get("name"))
        if name in CUT_NAMES:
            by_name[name] = CutSpec(name, int(cut.get("valid_time")), int(cut.get("tx_time")))
    if tuple(by_name) != CUT_NAMES and set(by_name) != set(CUT_NAMES):
        return None
    tenant = _text(document.get("tenant_id") or "tenant")
    entity = _text(document.get("entity_key") or document.get("family_id"))
    records = tuple(_record_from_mapping(item, tenant, entity) for item in records_raw if isinstance(item, Mapping))
    if not records:
        return None
    labeled_synthetic = bool(document.get("synthetic", synthetic))
    if labeled_synthetic and _text(document.get("cohort_source")) == COHORT_WILD:
        raise StreamParseError("synthetic family cannot declare frozen_wild_snapshots")
    return AdjudicatedFamily(
        family_id=_text(document.get("family_id")),
        stratum=stratum,
        tenant_id=tenant,
        entity_key=entity,
        records=records,
        cuts=tuple(by_name[name] for name in CUT_NAMES),
        adjudicated=True,
        synthetic=labeled_synthetic,
    )


def _protected_file(path: Path) -> bool:
    return any(
        "sealed_key" in part.casefold() or "blind-followup" in part.casefold() or part.casefold() == PROTECTED_NAME_PARTS[2]
        for part in path.parts
    )


def load_adjudicated_families(directory: Path) -> Tuple[AdjudicatedFamily, ...]:
    """Load adjudicated family envelopes from a snapshot directory."""
    if not directory.is_dir():
        return ()
    families = []
    for path in sorted(directory.glob("*.json")):
        if _protected_file(path):
            continue
        payload = load_json_snapshot(path)
        documents = payload if isinstance(payload, list) else [payload]
        for document in documents:
            if isinstance(document, Mapping):
                family = parse_family_envelope(document)
                if family is not None:
                    families.append(family)
    return tuple(families)


def count_adjudicated_families(families: Sequence[AdjudicatedFamily]) -> dict:
    """Count real adjudicated families. Synthetic rows do not count toward 240."""
    real = [family for family in families if family.adjudicated and not family.synthetic]
    by_stratum = {stratum: 0 for stratum in STRATA}
    for family in real:
        by_stratum[family.stratum] += 1
    return {
        "total": len(real),
        "by_stratum": by_stratum,
        "meets_wild_gate": len(real) >= WILD_FAMILY_MINIMUM and all(count >= WILD_PER_STRATUM for count in by_stratum.values()),
    }


def echo_decision(text_a: str, text_b: str, tenant_a: str, tenant_b: str) -> bool:
    """Collapse same-tenant paraphrases. Cross-tenant pairs never collapse."""
    if tenant_a != tenant_b:
        return False
    action, _reason = classify_high_similarity_pair(text_a, text_b)
    return action == "ECHO_COLLAPSE"


def unsigned_correlation_collapse(text_a: str, text_b: str) -> bool:
    """Baseline that ignores tenant identity and can leak across tenants."""
    action, _reason = classify_high_similarity_pair(text_a, text_b)
    return action == "ECHO_COLLAPSE"


def infer_echo_roots(records: Sequence[StreamRecord]) -> dict:
    """Cluster same-tenant paraphrases. The earliest record is the only root."""
    assertions = [record for record in records if record.op_kind in ASSERTING and record.text]
    parent = {record.record_id: record.record_id for record in assertions}

    def find(record_id: str) -> str:
        while parent[record_id] != record_id:
            parent[record_id] = parent[parent[record_id]]
            record_id = parent[record_id]
        return record_id

    def unite(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(assertions):
        for right in assertions[index + 1 :]:
            if echo_decision(left.text, right.text, left.tenant_id, right.tenant_id):
                unite(left.record_id, right.record_id)
    grouped = defaultdict(list)
    for record in assertions:
        grouped[find(record.record_id)].append(record)
    roots = {}
    for members in grouped.values():
        root = min(members, key=lambda record: (record.tx_time, record.record_id))
        for record in members:
            roots[record.record_id] = root.record_id
    return roots


def echo_metrics(pairs: Sequence[EchoPair], *, predicate) -> dict:
    """Precision and recall of a collapse predicate, plus cross-tenant leaks."""
    true_positive = false_positive = false_negative = true_negative = leaks = 0
    for pair in pairs:
        predicted = bool(predicate(pair))
        if pair.cross_tenant and predicted:
            leaks += 1
        if predicted and pair.expect_collapse:
            true_positive += 1
        elif predicted and not pair.expect_collapse:
            false_positive += 1
        elif pair.expect_collapse:
            false_negative += 1
        else:
            true_negative += 1
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    return {
        "precision": 1.0 if predicted_positive == 0 else true_positive / predicted_positive,
        "recall": 1.0 if actual_positive == 0 else true_positive / actual_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "cross_tenant_collapse_leaks": leaks,
    }


def _covers(record: StreamRecord, valid_time: int, tx_time: int) -> bool:
    if record.tx_time > tx_time:
        return False
    if record.valid_from > valid_time:
        return False
    if record.valid_to is not None and valid_time >= record.valid_to:
        return False
    return True


def _status_ops(records: Sequence[StreamRecord], tx_time: int) -> dict:
    """Latest effective WITHDRAW or REINSTATE per target at tx_time."""
    chosen: dict = {}
    for record in records:
        if record.op_kind not in {"WITHDRAW", "REINSTATE"} or record.tx_time > tx_time:
            continue
        target = record.target_id or record.record_id
        previous = chosen.get(target)
        if previous is None or (record.tx_time, record.record_id) >= (previous.tx_time, previous.record_id):
            chosen[target] = record
    return chosen


def oracle_at_cut(family: AdjudicatedFamily, cut: CutSpec) -> CutOracle:
    """Independent disposition and lineage at one frozen ``(valid_time, tx_time)``.

    Echoes and explicit derivation parents add no corroboration. A mirror cycle
    with no external root contributes nothing. Equal-authority incompatible
    claims are Disputed. A superseded filing is not current. Routine amendments
    do not become non-reliance unless a true Item 4.02 claim is itself current.
    """
    status_ops = _status_ops(family.records, cut.tx_time)
    assertions = []
    for record in family.records:
        if record.op_kind not in ASSERTING or record.tenant_id != family.tenant_id:
            continue
        if record.entity_key != family.entity_key or not _covers(record, cut.valid_time, cut.tx_time):
            continue
        op = status_ops.get(record.record_id)
        if op is not None and op.authority_rank >= record.authority_rank and op.op_kind == "WITHDRAW":
            continue
        assertions.append(record)
    by_id = {record.record_id: record for record in assertions}
    superseded = set()
    changed = True
    while changed:
        changed = False
        for record in assertions:
            if record.record_id in superseded:
                continue
            if any(target in by_id and target not in superseded for target in record.supersedes_ids):
                for target in record.supersedes_ids:
                    if target in by_id and target not in superseded:
                        superseded.add(target)
                        changed = True
    current = [record for record in assertions if record.record_id not in superseded]

    def supported(record: StreamRecord, seen: Tuple[str, ...]) -> bool:
        if record.record_id in seen:
            return False
        parents = tuple(record.explicit_parent_ids)
        if record.echo_of:
            parents = parents + (record.echo_of,)
        if not parents:
            return True
        for parent_id in parents:
            parent = by_id.get(parent_id)
            if parent is None or parent.record_id in superseded:
                return False
            if not supported(parent, seen + (record.record_id,)):
                return False
        return False

    roots = []
    for record in current:
        linked = bool(record.echo_of or record.explicit_parent_ids)
        if linked:
            continue
        if supported(record, ()):
            roots.append(record)
    if not roots:
        disposition = "retracted"
        lineage: Tuple[str, ...] = ()
    else:
        top = max(record.authority_rank for record in roots)
        winners = [record for record in roots if record.authority_rank == top]
        claims = {record.claim_value for record in winners}
        if len(claims) > 1 or any(record.op_kind == "CONFLICT" for record in winners):
            disposition = "disputed"
            lineage = tuple(sorted(record.record_id for record in winners))
        else:
            disposition = winners[0].claim_value
            lineage = tuple(sorted(record.record_id for record in winners))
    return CutOracle(cut.name, cut.valid_time, cut.tx_time, disposition, lineage)


def family_oracles(family: AdjudicatedFamily) -> Tuple[CutOracle, ...]:
    return tuple(oracle_at_cut(family, cut) for cut in family.cuts)


def _cut(name: str, tick: int) -> CutSpec:
    return CutSpec(name, tick, tick)


def _assert(
    record_id: str,
    tenant: str,
    entity: str,
    claim: str,
    tick: int,
    text: str,
    source: str,
    *,
    echo_of: str = "",
    supersedes: Tuple[str, ...] = (),
    form: str = "",
    items: Tuple[str, ...] = (),
    authority: int = 5,
) -> StreamRecord:
    return StreamRecord(
        record_id=record_id,
        tenant_id=tenant,
        entity_key=entity,
        claim_value=claim,
        authority_rank=authority,
        valid_from=tick,
        valid_to=None,
        tx_time=tick,
        op_kind="ASSERT",
        text=text,
        source=source,
        echo_of=echo_of,
        supersedes_ids=supersedes,
        form=form,
        items=items,
    )


def _withdraw(record_id: str, target: str, tenant: str, entity: str, tick: int) -> StreamRecord:
    return StreamRecord(
        record_id=record_id,
        tenant_id=tenant,
        entity_key=entity,
        claim_value="",
        authority_rank=5,
        valid_from=tick,
        valid_to=None,
        tx_time=tick,
        op_kind="WITHDRAW",
        text="",
        source="status",
        target_id=target,
    )


def build_schema_drycheck_families() -> Tuple[AdjudicatedFamily, ...]:
    """One synthetic family per required stratum. These records are not wild."""
    echo_tenant = "tenant-echo"
    echo_entity = "family-echo"
    conflict_tenant = "tenant-conflict"
    conflict_entity = "family-conflict"
    restatement_tenant = "tenant-restatement"
    restatement_entity = "family-restatement"
    amend_tenant = "tenant-amend"
    amend_entity = "family-amend"
    families = (
        AdjudicatedFamily(
            family_id="dry-echo",
            stratum="unlinked_cross_mirror_echo",
            tenant_id=echo_tenant,
            entity_key=echo_entity,
            synthetic=True,
            records=(
                _assert("echo-origin", echo_tenant, echo_entity, "affected", 10, ORIGIN_TEXT, "ghsa"),
                _assert("echo-mirror-a", echo_tenant, echo_entity, "affected", 20, ECHO_TEXT, "nvd", echo_of="echo-origin"),
                _assert("echo-mirror-b", echo_tenant, echo_entity, "affected", 20, ECHO_TEXT_2, "distro", echo_of="echo-origin"),
                _assert(
                    "echo-cross-tenant",
                    "tenant-other",
                    "family-other",
                    "affected",
                    20,
                    ORIGIN_TEXT,
                    "downstream",
                ),
                _withdraw("echo-withdraw", "echo-origin", echo_tenant, echo_entity, 30),
                _assert("echo-reinstated", echo_tenant, echo_entity, "affected", 40, REINSTATE_TEXT, "ghsa"),
            ),
            cuts=(_cut("t_initial", 10), _cut("t_mirror_echo_or_conflict", 20), _cut("t_withdrawal_or_402", 30), _cut("t_reinstatement_or_restated_10ka", 40)),
        ),
        AdjudicatedFamily(
            family_id="dry-conflict",
            stratum="equal_authority_vex_ghsa",
            tenant_id=conflict_tenant,
            entity_key=conflict_entity,
            synthetic=True,
            records=(
                _assert("conflict-ghsa", conflict_tenant, conflict_entity, "affected", 10, GHSA_TEXT, "ghsa", form="", items=()),
                _assert("conflict-vex", conflict_tenant, conflict_entity, "not_affected", 20, VEX_TEXT, "csaf_vex"),
                _withdraw("conflict-withdraw", "conflict-ghsa", conflict_tenant, conflict_entity, 30),
                _assert("conflict-ghsa-restored", conflict_tenant, conflict_entity, "affected", 40, GHSA_RESTORED_TEXT, "ghsa"),
            ),
            cuts=(_cut("t_initial", 10), _cut("t_mirror_echo_or_conflict", 20), _cut("t_withdrawal_or_402", 30), _cut("t_reinstatement_or_restated_10ka", 40)),
        ),
        AdjudicatedFamily(
            family_id="dry-402",
            stratum="sec_item_402_nonreliance",
            tenant_id=restatement_tenant,
            entity_key=restatement_entity,
            synthetic=True,
            records=(
                _assert("sec-10k", restatement_tenant, restatement_entity, "reliance_unchanged", 10, TENK_TEXT, "sec", form="10-K"),
                _assert(
                    "sec-press",
                    restatement_tenant,
                    restatement_entity,
                    "reliance_unchanged",
                    20,
                    PRESS_TEXT,
                    "wire",
                    echo_of="sec-10k",
                ),
                _assert(
                    "sec-8k-402",
                    restatement_tenant,
                    restatement_entity,
                    "non_reliance",
                    30,
                    ITEM_402_TEXT,
                    "sec",
                    supersedes=("sec-10k",),
                    form="8-K",
                    items=("4.02",),
                ),
                _assert(
                    "sec-restated-10ka",
                    restatement_tenant,
                    restatement_entity,
                    "restated_current",
                    40,
                    RESTATED_TEXT,
                    "sec",
                    supersedes=("sec-8k-402",),
                    form="10-K/A",
                ),
            ),
            cuts=(_cut("t_initial", 10), _cut("t_mirror_echo_or_conflict", 20), _cut("t_withdrawal_or_402", 30), _cut("t_reinstatement_or_restated_10ka", 40)),
        ),
        AdjudicatedFamily(
            family_id="dry-amend",
            stratum="routine_amendment_control",
            tenant_id=amend_tenant,
            entity_key=amend_entity,
            synthetic=True,
            records=(
                _assert("amend-10k", amend_tenant, amend_entity, "reliance_unchanged", 10, TENK_TEXT, "sec", form="10-K"),
                _assert(
                    "amend-press",
                    amend_tenant,
                    amend_entity,
                    "reliance_unchanged",
                    20,
                    PRESS_TEXT,
                    "wire",
                    echo_of="amend-10k",
                ),
                _assert(
                    "amend-10ka",
                    amend_tenant,
                    amend_entity,
                    "reliance_unchanged",
                    30,
                    AMEND_TEXT,
                    "sec",
                    supersedes=("amend-10k",),
                    form="10-K/A",
                ),
                _assert(
                    "amend-8ka",
                    amend_tenant,
                    amend_entity,
                    "reliance_unchanged",
                    40,
                    AMEND_FOLLOWUP_TEXT,
                    "sec",
                    supersedes=("amend-10ka",),
                    form="8-K/A",
                    items=("5.02",),
                ),
            ),
            cuts=(_cut("t_initial", 10), _cut("t_mirror_echo_or_conflict", 20), _cut("t_withdrawal_or_402", 30), _cut("t_reinstatement_or_restated_10ka", 40)),
        ),
    )
    for family in families:
        if not family.synthetic:
            raise StreamParseError("schema drycheck families must stay synthetic")
    return families


def schema_echo_pairs(families: Sequence[AdjudicatedFamily] = ()) -> Tuple[EchoPair, ...]:
    """Labeled collapse pairs, including cross-tenant near-copies that must not collapse."""
    families = families or build_schema_drycheck_families()
    pairs = [
        EchoPair(ORIGIN_TEXT, ECHO_TEXT, "tenant-echo", "tenant-echo", True),
        EchoPair(ORIGIN_TEXT, ECHO_TEXT_2, "tenant-echo", "tenant-echo", True),
        EchoPair(ORIGIN_TEXT, ORIGIN_TEXT, "tenant-echo", "tenant-other", False),
        EchoPair(GHSA_TEXT, VEX_TEXT, "tenant-conflict", "tenant-conflict", False),
        EchoPair(TENK_TEXT, PRESS_TEXT, "tenant-restatement", "tenant-restatement", True),
        EchoPair(TENK_TEXT, ITEM_402_TEXT, "tenant-restatement", "tenant-restatement", False),
        EchoPair(TENK_TEXT, RESTATED_TEXT, "tenant-restatement", "tenant-restatement", False),
        EchoPair(TENK_TEXT, AMEND_TEXT, "tenant-amend", "tenant-amend", False),
        EchoPair(AMEND_TEXT, AMEND_FOLLOWUP_TEXT, "tenant-amend", "tenant-amend", False),
    ]
    for family in families:
        assertions = [record for record in family.records if record.op_kind in ASSERTING and record.text]
        for left in assertions:
            for right in assertions:
                if left.record_id >= right.record_id:
                    continue
                same_tenant = left.tenant_id == right.tenant_id
                shared_root = bool(left.echo_of and right.echo_of and left.echo_of == right.echo_of)
                points_at_other = bool(
                    (left.echo_of and left.echo_of == right.record_id) or (right.echo_of and right.echo_of == left.record_id)
                )
                expect = same_tenant and (shared_root or points_at_other)
                if left.echo_of or right.echo_of or not same_tenant:
                    pairs.append(
                        EchoPair(left.text, right.text, left.tenant_id, right.tenant_id, expect)
                    )
    return tuple(pairs)


def cycle_without_external_root() -> AdjudicatedFamily:
    """Two explicit parents that only cite each other. Neither is corroboration."""
    tenant, entity = "tenant-cycle", "family-cycle"
    left = StreamRecord(
        record_id="cycle-a",
        tenant_id=tenant,
        entity_key=entity,
        claim_value="affected",
        authority_rank=5,
        valid_from=10,
        valid_to=None,
        tx_time=10,
        op_kind="ASSERT",
        text="Cycle member A cites only its mirror partner.",
        source="ghsa",
        explicit_parent_ids=("cycle-b",),
    )
    right = StreamRecord(
        record_id="cycle-b",
        tenant_id=tenant,
        entity_key=entity,
        claim_value="affected",
        authority_rank=5,
        valid_from=10,
        valid_to=None,
        tx_time=10,
        op_kind="ASSERT",
        text="Cycle member B cites only its mirror partner.",
        source="nvd",
        explicit_parent_ids=("cycle-a",),
    )
    return AdjudicatedFamily(
        family_id="dry-cycle",
        stratum="unlinked_cross_mirror_echo",
        tenant_id=tenant,
        entity_key=entity,
        synthetic=True,
        records=(left, right),
        cuts=(_cut("t_initial", 10), _cut("t_mirror_echo_or_conflict", 10), _cut("t_withdrawal_or_402", 10), _cut("t_reinstatement_or_restated_10ka", 10)),
    )


def select_cohort(wild_families: Sequence[AdjudicatedFamily]) -> dict:
    """Apply the sparse-stratum rule. Synthetic rows never fill a wild quota."""
    census = count_adjudicated_families(wild_families)
    if census["meets_wild_gate"]:
        selected = tuple(family for family in wild_families if family.adjudicated and not family.synthetic)
        return {
            "status": "EVALUATED",
            "cohort_source": COHORT_WILD,
            "families": selected,
            "descriptive_families": (),
            "census": census,
            "synthetic_labeled_wild": 0,
        }
    descriptive = tuple(family for family in wild_families if family.adjudicated and not family.synthetic)
    return {
        "status": STATUS_DRYCHECK,
        "cohort_source": COHORT_DRYCHECK,
        "families": build_schema_drycheck_families(),
        "descriptive_families": descriptive,
        "descriptive_cohort_source": COHORT_DESCRIPTIVE if descriptive else "",
        "census": census,
        "synthetic_labeled_wild": 0,
    }
