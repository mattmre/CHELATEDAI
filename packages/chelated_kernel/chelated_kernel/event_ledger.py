"""Stratified 3-valued ledger with integer Ed25519 hash-chained receipts.

Envelopes: CloudEvents 1.0, W3C PROV, OSV, CSAF VEX, Git commits/symbols, and CDC.
Truth values are current, withdrawn, and disputed. No write path calls a model.
non_nda certificates are decimal integers plus signature hex so they can be
scanned against the 56-pattern non-NDA denylist.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from chelated_kernel.errors import EnvelopeError, ReceiptVerificationError, TenantIsolationError
from chelated_kernel.profiles import LedgerPolicy, get_ledger_policy, require_tenant_id

OPEN_END = 1 << 62
INT64_MIN = -9223372036854775808
INT64_MAX = 9223372036854775807

RECEIPT_SPEC: tuple[str, ...] = (
    "schema_version",
    "bundle_id",
    "policy_code",
    "disclosure_code",
    "tenant_0",
    "tenant_1",
    "tenant_2",
    "tenant_3",
    "chain_index",
    "ingest_count",
    "current_count",
    "withdrawn_count",
    "disputed_count",
    "echo_cluster_count",
    "echo_member_count",
    "audit_count",
    "audit_surviving",
    "erasure_count",
    "visible_count",
    "snapshot_valid_time",
    "snapshot_tx_time",
    "min_valid_from",
    "max_valid_from",
    "min_tx_from",
    "max_tx_from",
    "prev_0",
    "prev_1",
    "prev_2",
    "prev_3",
    "state_0",
    "state_1",
    "state_2",
    "state_3",
    "claim_count",
    "edge_count",
    "llm_write_tokens",
    "cross_tenant_reads",
    "ceiling_blocks",
    "wfs_rounds",
    "stratum0_visible",
    "unfounded_count",
    "ingress_cloudevent",
    "ingress_prov",
    "ingress_osv",
    "ingress_csaf",
    "ingress_git",
    "ingress_cdc",
    "ingress_stream",
    "profile_code",
    "reserved_register_hits",
    "collapsed_support_ignored",
    "gdpr_redactions",
    "chain_domain",
    "field_count",
    "ip_dirty_fields",
    "zero_pad",
)

if len(RECEIPT_SPEC) != 56:
    raise RuntimeError(f"receipt schema width {len(RECEIPT_SPEC)} != 56")

CONTENT_BEARING_FIELDS: frozenset[str] = frozenset()


@dataclass
class Claim:
    claim_id: str
    kind: str
    status: str
    echo_key: str | None
    conflict_key: str | None
    authority: int
    base: bool
    audit_only: bool
    reserved_disputed: bool
    valid_from: int
    valid_to: int
    tx_from: int
    tx_to: int
    payload: dict[str, object] | None
    op: str | None = None
    pk: str | None = None
    table: str | None = None
    family: str = "ce"


@dataclass
class Edge:
    src: str
    dst: str
    kind: str
    authority: int
    in_dv: bool
    valid_from: int
    valid_to: int
    tx_from: int
    tx_to: int


@dataclass(frozen=True)
class AuditReceipt:
    integers: tuple[int, ...]
    signature: bytes
    public_key: bytes
    prev_hash: bytes
    body_hash: bytes
    receipt_hash: bytes

    def canonical_bytes(self) -> bytes:
        if len(self.integers) != 56:
            raise ReceiptVerificationError("receipt body is not 56 integers")
        return struct.pack("<56q", *self.integers)


@dataclass(frozen=True)
class IngestResult:
    claim_ids: tuple[str, ...]
    tx_time: int
    receipt_hash: str


@dataclass(frozen=True)
class EvalStats:
    truth: dict[str, str]
    ceiling_blocks: int
    wfs_rounds: int
    unfounded_count: int
    visible_count: int
    echo_cluster_count: int
    echo_member_count: int
    collapsed_support_ignored: int
    independent: dict[str, bool]


def _i64(value: int) -> int:
    number = int(value)
    if number < INT64_MIN or number > INT64_MAX:
        raise ReceiptVerificationError("receipt integer outside int64")
    return number


def _digest_i64(digest: bytes) -> tuple[int, int, int, int]:
    return struct.unpack("<4q", digest[:32])


def _parse_time(value: object, fallback: int) -> int:
    if value is None:
        return int(fallback)
    if isinstance(value, bool):
        raise EnvelopeError("time", "boolean is not a timestamp")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        iso = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return delta.days * 86_400 * 1_000_000 + delta.seconds * 1_000_000 + delta.microseconds
    raise EnvelopeError("time", f"unsupported time {value!r}")


def _visible(valid_from: int, valid_to: int, tx_from: int, tx_to: int, valid_time: int, tx_time: int) -> bool:
    return valid_from <= valid_time < valid_to and tx_from <= tx_time < tx_to


def _canon_payload(payload: dict[str, object] | None) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _tombstone() -> dict[str, object]:
    return {"erased": 1}


class EventLedger:
    """One tenant, one policy, zero model write tokens."""

    def __init__(self, *, policy: str = "sec_audit", tenant_id: str = "default", disclosure: str = "non_nda") -> None:
        self.policy: LedgerPolicy = get_ledger_policy(policy)
        self.tenant_id = require_tenant_id(tenant_id)
        if disclosure not in {"non_nda", "internal"}:
            raise EnvelopeError("disclosure", "disclosure must be non_nda or internal")
        self.disclosure = disclosure
        self.llm_write_tokens = 0
        self.unauthorized_reads = 0
        self.gdpr_redactions = 0
        self._tx = 0
        self._claims: dict[str, Claim] = {}
        self._edges: list[Edge] = []
        self._receipts: list[AuditReceipt] = []
        self._ingress = {"ce": 0, "prov": 0, "osv": 0, "csaf": 0, "git": 0, "cdc": 0, "stream": 0}
        self._reserved_hits = 0
        seed = hashlib.sha256(
            f"chelated-kernel-ed25519-v1|{self.tenant_id}|{self.policy.name}|{self.disclosure}".encode("utf-8")
        ).digest()
        self._private = Ed25519PrivateKey.from_private_bytes(seed)
        self._public = self._private.public_key()
        self._public_bytes = self._public.public_bytes_raw()

    @property
    def receipts(self) -> tuple[AuditReceipt, ...]:
        return tuple(self._receipts)

    def ingest(self, envelope: dict[str, object]) -> IngestResult:
        if not isinstance(envelope, dict):
            raise EnvelopeError("envelope", "envelope must be a dict")
        foreign = envelope.get("tenant_id")
        if foreign is not None and str(foreign) != self.tenant_id:
            raise TenantIsolationError("envelope tenant_id does not match this ledger")
        self._tx += 1
        tx_time = self._tx
        claims, edges, family = _parse_envelope(envelope, authority=self.policy.default_authority, tx_hint=tx_time)
        if family not in self._ingress:
            raise EnvelopeError("family", f"unknown ingress family {family}")
        if self.llm_write_tokens != 0:
            raise ReceiptVerificationError("write path recorded a model token")
        self._ingress[family] = self._ingress[family] + 1
        stored: list[Claim] = []
        renamed: dict[str, str] = {}
        for claim in claims:
            claim.tx_from = tx_time
            original_id = claim.claim_id
            if claim.claim_id in self._claims or claim.claim_id in renamed.values():
                claim.claim_id = f"{original_id}#{tx_time}"
                renamed[original_id] = claim.claim_id
            if claim.reserved_disputed:
                self._reserved_hits += 1
            self._claims[claim.claim_id] = claim
            stored.append(claim)
        for edge in edges:
            edge.tx_from = tx_time
            edge.src = renamed.get(edge.src, edge.src)
            edge.dst = renamed.get(edge.dst, edge.dst)
            self._edges.append(edge)
        self._link_status_conflicts(stored, tx_time)
        self._link_cdc(stored, tx_time)
        if self.policy.redact_on_erase:
            self._scrub_erased(stored)
        receipt = self._append_receipt(tx_time, stored)
        return IngestResult(
            claim_ids=tuple(claim.claim_id for claim in stored),
            tx_time=tx_time,
            receipt_hash=receipt.receipt_hash.hex(),
        )

    def as_of(self, valid_time: int, tx_time: int) -> list[dict[str, object]]:
        stats = self._evaluate(int(valid_time), int(tx_time))
        rows: list[dict[str, object]] = []
        for claim in sorted(self._claims.values(), key=lambda item: item.claim_id):
            if not _visible(claim.valid_from, claim.valid_to, claim.tx_from, claim.tx_to, int(valid_time), int(tx_time)):
                continue
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "kind": claim.kind,
                    "status": claim.status,
                    "truth": stats.truth.get(claim.claim_id, "withdrawn"),
                    "audit_only": claim.audit_only,
                    "collapsed": not stats.independent.get(claim.claim_id, True),
                    "independent": stats.independent.get(claim.claim_id, True),
                    "valid_from": claim.valid_from,
                    "tx_from": claim.tx_from,
                    "payload": claim.payload,
                }
            )
        return rows

    def echo_clusters(self, valid_time: int, tx_time: int) -> list[tuple[str, ...]]:
        return _unlinked_clusters(self._visible_claims(int(valid_time), int(tx_time)), self._visible_edges(int(valid_time), int(tx_time)))

    def verify_chain(self) -> int:
        prev = bytes(32)
        for index, receipt in enumerate(self._receipts):
            _verify_one(receipt, public_key=self._public, prev_hash=prev)
            if receipt.prev_hash != prev:
                raise ReceiptVerificationError(f"chain break at {index}")
            prev = receipt.receipt_hash
        return len(self._receipts)

    def non_nda_certificate(self) -> str:
        if self.disclosure != "non_nda":
            raise ReceiptVerificationError("certificate export requires non_nda disclosure")
        lines: list[str] = []
        for receipt in self._receipts:
            lines.append(",".join(str(number) for number in receipt.integers))
            lines.append(receipt.signature.hex())
            lines.append(receipt.receipt_hash.hex())
        return "\n".join(lines)

    def non_nda_ip_report(self) -> dict[str, int | str]:
        dirty = 0
        for receipt in self._receipts:
            if len(receipt.integers) != 56:
                dirty += 56
                continue
            for name, value in zip(RECEIPT_SPEC, receipt.integers):
                if name in CONTENT_BEARING_FIELDS or not isinstance(value, int):
                    dirty += 1
        text = self.non_nda_certificate() if self.disclosure == "non_nda" else ""
        payload_leaks = 0
        for token in text.replace(",", " ").replace("\n", " ").split():
            bare = token[1:] if token.startswith("-") else token
            if bare and not all(char in "0123456789abcdef" for char in bare):
                dirty += 1
                payload_leaks += 1
        for claim in self._claims.values():
            encoded = _canon_payload(claim.payload)
            if encoded and encoded in text:
                payload_leaks += 1
        return {
            "dirty_fields": dirty,
            "checked_fields": 56,
            "payload_leaks": payload_leaks,
            "ratio": f"{dirty}/56",
        }

    def _visible_claims(self, valid_time: int, tx_time: int) -> list[Claim]:
        return [
            claim
            for claim in self._claims.values()
            if _visible(claim.valid_from, claim.valid_to, claim.tx_from, claim.tx_to, valid_time, tx_time)
        ]

    def _visible_edges(self, valid_time: int, tx_time: int) -> list[Edge]:
        return [
            edge
            for edge in self._edges
            if _visible(edge.valid_from, edge.valid_to, edge.tx_from, edge.tx_to, valid_time, tx_time)
        ]

    def _link_status_conflicts(self, fresh: list[Claim], tx_time: int) -> None:
        for claim in fresh:
            if claim.conflict_key is None or claim.status not in {"affected", "not_affected"}:
                continue
            for old in list(self._claims.values()):
                if old.claim_id == claim.claim_id or old.conflict_key != claim.conflict_key:
                    continue
                if old.status not in {"affected", "not_affected"} or old.status == claim.status:
                    continue
                kind = "conflicts" if old.authority == claim.authority else "supersede"
                src, dst = (claim.claim_id, old.claim_id) if claim.authority >= old.authority else (old.claim_id, claim.claim_id)
                self._edges.append(
                    Edge(src, dst, kind, max(claim.authority, old.authority), True, claim.valid_from, OPEN_END, tx_time, OPEN_END)
                )
                if kind == "conflicts":
                    self._edges.append(
                        Edge(dst, src, "conflicts", claim.authority, True, claim.valid_from, OPEN_END, tx_time, OPEN_END)
                    )

    def _link_cdc(self, fresh: list[Claim], tx_time: int) -> None:
        for claim in fresh:
            if claim.kind != "cdc_row" or claim.op not in {"u", "d"}:
                continue
            for old in list(self._claims.values()):
                if old.claim_id == claim.claim_id or old.kind != "cdc_row":
                    continue
                if old.table != claim.table or old.pk != claim.pk or old.tx_from > tx_time:
                    continue
                kind = "erases" if claim.op == "d" else "supersede"
                self._edges.append(
                    Edge(claim.claim_id, old.claim_id, kind, claim.authority, True, claim.valid_from, OPEN_END, tx_time, OPEN_END)
                )

    def _scrub_erased(self, fresh: list[Claim]) -> None:
        targets = {edge.dst for edge in self._edges if edge.kind == "erases" and edge.src in {claim.claim_id for claim in fresh}}
        if not targets and not any(claim.op == "d" for claim in fresh):
            return
        for claim in fresh:
            if claim.op == "d":
                claim.payload = _tombstone()
                self.gdpr_redactions += 1
        for claim_id in targets:
            claim = self._claims.get(claim_id)
            if claim is None:
                continue
            claim.payload = _tombstone()
            self.gdpr_redactions += 1

    def _evaluate(self, valid_time: int, tx_time: int) -> EvalStats:
        claims = self._visible_claims(valid_time, tx_time)
        edges = self._visible_edges(valid_time, tx_time)
        clusters = _unlinked_clusters(claims, edges)
        representatives: dict[str, str] = {}
        independent = {claim.claim_id: True for claim in claims}
        echo_members = 0
        for members in clusters:
            ordered = sorted(members, key=lambda claim: (claim.tx_from, claim.claim_id))
            root = ordered[0]
            for claim in ordered[1:]:
                representatives[claim.claim_id] = root.claim_id
                independent[claim.claim_id] = False
                echo_members += 1
            independent[root.claim_id] = True
        by_id = {claim.claim_id: claim for claim in claims}
        attacks: dict[str, list[Edge]] = {claim.claim_id: [] for claim in claims}
        supports: dict[str, list[Edge]] = {claim.claim_id: [] for claim in claims}
        ceiling = 0
        collapsed_ignored = 0
        for edge in edges:
            if edge.src not in by_id or edge.dst not in by_id:
                continue
            if not edge.in_dv or edge.kind == "audit_citation":
                continue
            if edge.src in representatives:
                collapsed_ignored += 1
                continue
            src = by_id[edge.src]
            dst = by_id[edge.dst]
            if edge.kind in {"supersede", "refute", "erases"}:
                if src.authority < dst.authority:
                    ceiling += 1
                    continue
                attacks[edge.dst].append(edge)
            elif edge.kind == "conflicts" and src.authority == dst.authority:
                attacks[edge.dst].append(edge)
            elif edge.kind == "supports":
                supports[edge.dst].append(edge)
        conflicted = set()
        for claim in claims:
            if claim.reserved_disputed:
                conflicted.add(claim.claim_id)
            for edge in attacks[claim.claim_id]:
                if edge.kind == "conflicts":
                    conflicted.add(claim.claim_id)
                    conflicted.add(edge.src)
        state = {claim.claim_id: "undefined" for claim in claims if claim.claim_id not in representatives}
        rounds = 0
        for _round in range(len(state) + 2):
            rounds += 1
            nxt: dict[str, str] = {}
            for claim_id, current in state.items():
                claim = by_id[claim_id]
                if claim_id in conflicted:
                    nxt[claim_id] = "disputed"
                    continue
                if claim.audit_only:
                    nxt[claim_id] = "current"
                    continue
                defeated = False
                for edge in attacks[claim_id]:
                    if edge.kind in {"supersede", "refute", "erases"} and state.get(edge.src) == "current":
                        defeated = True
                if defeated:
                    nxt[claim_id] = "withdrawn"
                    continue
                if claim.base:
                    nxt[claim_id] = "current"
                    continue
                support_states = [state.get(edge.src, "undefined") for edge in supports[claim_id]]
                if any(item == "current" for item in support_states):
                    nxt[claim_id] = "current"
                elif any(item == "undefined" for item in support_states):
                    nxt[claim_id] = "undefined"
                else:
                    nxt[claim_id] = "withdrawn"
            if nxt == state:
                break
            state = nxt
        unfounded = 0
        for claim_id, value in list(state.items()):
            if value == "undefined":
                state[claim_id] = "withdrawn"
                if not by_id[claim_id].base:
                    unfounded += 1
        truth: dict[str, str] = {}
        for claim in claims:
            root = representatives.get(claim.claim_id, claim.claim_id)
            truth[claim.claim_id] = state.get(root, "withdrawn")
        return EvalStats(
            truth=truth,
            ceiling_blocks=ceiling,
            wfs_rounds=rounds,
            unfounded_count=unfounded,
            visible_count=len(claims),
            echo_cluster_count=len(clusters),
            echo_member_count=echo_members,
            collapsed_support_ignored=collapsed_ignored,
            independent=independent,
        )

    def _append_receipt(self, tx_time: int, fresh: list[Claim]) -> AuditReceipt:
        valid_time = max((claim.valid_from for claim in fresh), default=tx_time)
        stats = self._evaluate(valid_time, tx_time)
        counts = {"current": 0, "withdrawn": 0, "disputed": 0}
        audit_count = 0
        audit_surviving = 0
        erasure_count = 0
        min_valid = OPEN_END
        max_valid = 0
        min_tx = OPEN_END
        max_tx = 0
        for claim in self._visible_claims(valid_time, tx_time):
            label = stats.truth.get(claim.claim_id, "withdrawn")
            if stats.independent.get(claim.claim_id, True):
                counts[label] = counts.get(label, 0) + 1
            if claim.audit_only:
                audit_count += 1
                if label == "current":
                    audit_surviving += 1
            if claim.op == "d" or claim.status == "erased":
                erasure_count += 1
            min_valid = min(min_valid, claim.valid_from)
            max_valid = max(max_valid, claim.valid_from)
            min_tx = min(min_tx, claim.tx_from)
            max_tx = max(max_tx, claim.tx_from)
        if min_valid == OPEN_END:
            min_valid = 0
            min_tx = 0
        prev = self._receipts[-1].receipt_hash if self._receipts else bytes(32)
        state_digest = self._state_digest(valid_time, tx_time, stats.truth)
        tenant_digest = hashlib.sha256(self.tenant_id.encode("utf-8")).digest()
        values = {
            "schema_version": 1,
            "bundle_id": 49,
            "policy_code": self.policy.code,
            "disclosure_code": 1 if self.disclosure == "non_nda" else 2,
            "tenant_0": _digest_i64(tenant_digest)[0],
            "tenant_1": _digest_i64(tenant_digest)[1],
            "tenant_2": _digest_i64(tenant_digest)[2],
            "tenant_3": _digest_i64(tenant_digest)[3],
            "chain_index": len(self._receipts),
            "ingest_count": self._tx,
            "current_count": counts["current"],
            "withdrawn_count": counts["withdrawn"],
            "disputed_count": counts["disputed"],
            "echo_cluster_count": stats.echo_cluster_count,
            "echo_member_count": stats.echo_member_count,
            "audit_count": audit_count,
            "audit_surviving": audit_surviving,
            "erasure_count": erasure_count,
            "visible_count": stats.visible_count,
            "snapshot_valid_time": valid_time,
            "snapshot_tx_time": tx_time,
            "min_valid_from": min_valid,
            "max_valid_from": max_valid,
            "min_tx_from": min_tx,
            "max_tx_from": max_tx,
            "prev_0": _digest_i64(prev)[0],
            "prev_1": _digest_i64(prev)[1],
            "prev_2": _digest_i64(prev)[2],
            "prev_3": _digest_i64(prev)[3],
            "state_0": _digest_i64(state_digest)[0],
            "state_1": _digest_i64(state_digest)[1],
            "state_2": _digest_i64(state_digest)[2],
            "state_3": _digest_i64(state_digest)[3],
            "claim_count": len(self._claims),
            "edge_count": len(self._edges),
            "llm_write_tokens": self.llm_write_tokens,
            "cross_tenant_reads": self.unauthorized_reads,
            "ceiling_blocks": stats.ceiling_blocks,
            "wfs_rounds": stats.wfs_rounds,
            "stratum0_visible": stats.visible_count,
            "unfounded_count": stats.unfounded_count,
            "ingress_cloudevent": self._ingress["ce"],
            "ingress_prov": self._ingress["prov"],
            "ingress_osv": self._ingress["osv"],
            "ingress_csaf": self._ingress["csaf"],
            "ingress_git": self._ingress["git"],
            "ingress_cdc": self._ingress["cdc"],
            "ingress_stream": self._ingress["stream"],
            "profile_code": self.policy.code,
            "reserved_register_hits": self._reserved_hits,
            "collapsed_support_ignored": stats.collapsed_support_ignored,
            "gdpr_redactions": self.gdpr_redactions,
            "chain_domain": 1,
            "field_count": 56,
            "ip_dirty_fields": len(CONTENT_BEARING_FIELDS),
            "zero_pad": 0,
        }
        integers = tuple(_i64(values[name]) for name in RECEIPT_SPEC)
        body = struct.pack("<56q", *integers)
        body_hash = hashlib.sha256(body).digest()
        signature = self._private.sign(prev + body_hash)
        receipt_hash = hashlib.sha256(signature + body_hash + prev).digest()
        receipt = AuditReceipt(
            integers=integers,
            signature=signature,
            public_key=self._public_bytes,
            prev_hash=prev,
            body_hash=body_hash,
            receipt_hash=receipt_hash,
        )
        self._receipts.append(receipt)
        return receipt

    def _state_digest(self, valid_time: int, tx_time: int, truth: dict[str, str]) -> bytes:
        lines: list[str] = []
        for claim in sorted(self._visible_claims(valid_time, tx_time), key=lambda item: item.claim_id):
            payload_hash = hashlib.sha256(_canon_payload(claim.payload).encode("utf-8")).hexdigest()
            lines.append(f"{claim.claim_id}|{truth.get(claim.claim_id, 'withdrawn')}|{payload_hash}")
        return hashlib.sha256("\n".join(lines).encode("utf-8")).digest()


def _verify_one(receipt: AuditReceipt, *, public_key: object, prev_hash: bytes) -> None:
    if len(receipt.integers) != 56:
        raise ReceiptVerificationError("receipt body is not 56 integers")
    body = receipt.canonical_bytes()
    body_hash = hashlib.sha256(body).digest()
    if body_hash != receipt.body_hash:
        raise ReceiptVerificationError("body hash mismatch")
    if receipt.prev_hash != prev_hash:
        raise ReceiptVerificationError("prev hash mismatch")
    expected = hashlib.sha256(receipt.signature + body_hash + prev_hash).digest()
    if expected != receipt.receipt_hash:
        raise ReceiptVerificationError("receipt hash mismatch")
    try:
        public_key.verify(receipt.signature, prev_hash + body_hash)  # type: ignore[attr-defined]
    except InvalidSignature as exc:
        raise ReceiptVerificationError("ed25519 signature rejected") from exc


def _unlinked_clusters(claims: list[Claim], edges: list[Edge]) -> list[list[Claim]]:
    grouped: dict[str, list[Claim]] = {}
    for claim in claims:
        if not claim.echo_key:
            continue
        grouped.setdefault(claim.echo_key, []).append(claim)
    linked: set[tuple[str, str]] = set()
    for edge in edges:
        if edge.kind in {"supports", "supersede", "refute", "erases", "conflicts", "audit_citation"}:
            linked.add((edge.src, edge.dst))
            linked.add((edge.dst, edge.src))
    clusters: list[list[Claim]] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        ids = [member.claim_id for member in members]
        if any((left, right) in linked for left in ids for right in ids if left != right):
            continue
        clusters.append(members)
    clusters.sort(key=lambda members: members[0].echo_key or "")
    return clusters


_PAYLOAD_DROP = frozenset(
    {
        "cites_audit",
        "CITES_AUDIT",
        "relation",
        "supersedes",
        "target",
        "refutes",
        "erases",
        "item",
        "register",
    }
)


def _clamp_authority(value: object, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EnvelopeError("authority", "authority must be an integer when it is present")
    if value < 0 or value > 100:
        raise EnvelopeError("authority", "authority must be inside 0..100")
    return int(value)


def _mapping(value: object, code: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise EnvelopeError(code, "a JSON object is required")
    return value


def _new_claim(
    claim_id: str,
    *,
    kind: str,
    status: str,
    authority: int,
    valid_from: int,
    base: bool,
    family: str,
    payload: dict[str, object] | None,
    echo_key: str | None = None,
    conflict_key: str | None = None,
    audit_only: bool = False,
    reserved_disputed: bool = False,
    op: str | None = None,
    pk: str | None = None,
    table: str | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        kind=kind,
        status=status,
        echo_key=echo_key,
        conflict_key=conflict_key,
        authority=authority,
        base=base,
        audit_only=audit_only,
        reserved_disputed=reserved_disputed,
        valid_from=valid_from,
        valid_to=OPEN_END,
        tx_from=0,
        tx_to=OPEN_END,
        payload=payload,
        op=op,
        pk=pk,
        table=table,
        family=family,
    )


def _new_edge(src: str, dst: str, kind: str, authority: int, in_dv: bool, valid_from: int) -> Edge:
    return Edge(src, dst, kind, authority, in_dv, valid_from, OPEN_END, 0, OPEN_END)


def _stored_payload(data: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in data.items() if key not in _PAYLOAD_DROP}


def _citation_target(envelope: dict[str, object], data: dict[str, object]) -> str | None:
    for source in (data, envelope):
        for key in ("CITES_AUDIT", "cites_audit"):
            if key not in source:
                continue
            value = source[key]
            if value is True:
                return ""
            if value is False or value is None:
                return None
            return str(value)
    relation = str(data.get("relation", envelope.get("relation", ""))).upper()
    if relation == "CITES_AUDIT":
        target = data.get("target", envelope.get("target"))
        return "" if target is None else str(target)
    return None


def _attack_edges(
    claim_id: str,
    envelope: dict[str, object],
    data: dict[str, object],
    authority: int,
    valid_from: int,
) -> list[Edge]:
    edges: list[Edge] = []
    relation = str(data.get("relation", envelope.get("relation", ""))).lower()
    supersedes = data.get("supersedes", envelope.get("supersedes"))
    refutes = data.get("refutes", envelope.get("refutes"))
    erases = data.get("erases", envelope.get("erases"))
    item = data.get("item", envelope.get("item"))
    item_text = str(item).strip() if item is not None else ""
    if item_text in {"4.02", "402", "item-4.02"}:
        if supersedes is None:
            raise EnvelopeError("sec_item_402", "item 4.02 requires supersedes")
        edges.append(_new_edge(claim_id, str(supersedes), "supersede", authority, True, valid_from))
    elif supersedes is not None or relation in {"supersede", "supercede"}:
        target = supersedes if supersedes is not None else data.get("target", envelope.get("target"))
        if target is None:
            raise EnvelopeError("supersede", "supersede relation requires a target")
        edges.append(_new_edge(claim_id, str(target), "supersede", authority, True, valid_from))
    elif refutes is not None or relation in {"refute", "refutes"}:
        target = refutes if refutes is not None else data.get("target", envelope.get("target"))
        if target is None:
            raise EnvelopeError("refute", "refute relation requires a target")
        edges.append(_new_edge(claim_id, str(target), "refute", authority, True, valid_from))
    elif erases is not None or relation in {"erase", "erases"}:
        target = erases if erases is not None else data.get("target", envelope.get("target"))
        if target is None:
            raise EnvelopeError("erases", "erasure relation requires a target")
        edges.append(_new_edge(claim_id, str(target), "erases", authority, True, valid_from))
    return edges


def _parse_cloudevent(
    envelope: dict[str, object],
    *,
    authority: int,
    tx_hint: int,
) -> tuple[list[Claim], list[Edge], str]:
    event_type = envelope.get("type")
    event_id = envelope.get("id")
    source = envelope.get("source")
    if not isinstance(event_type, str) or not event_type:
        raise EnvelopeError("cloudevent", "type is required")
    if not isinstance(event_id, str) or not event_id:
        raise EnvelopeError("cloudevent", "id is required")
    if not isinstance(source, str) or not source:
        raise EnvelopeError("cloudevent", "source is required")
    data = _mapping(envelope.get("data"), "data")
    chosen = _clamp_authority(envelope.get("authority", data.get("authority")), authority)
    valid_from = _parse_time(envelope.get("time", envelope.get("valid_time")), tx_hint)
    edges = _attack_edges(event_id, envelope, data, chosen, valid_from)
    cited = _citation_target(envelope, data)
    audit_only = cited is not None or event_type == "com.chelated.stream.suppression"
    if cited:
        edges.append(_new_edge(event_id, cited, "audit_citation", chosen, False, valid_from))
    register = data.get("register", envelope.get("register"))
    reserved = event_type == "com.chelated.dpu.flow" and register in {11, "11"}
    if event_type == "com.chelated.sec.8k":
        kind = "sec_restatement"
        status = "filed"
        family = "ce"
    elif event_type == "com.chelated.dpu.flow":
        kind = "dpu_flow"
        status = "observed"
        family = "ce"
    elif event_type == "com.chelated.agent.assertion":
        kind = "agent_assertion"
        status = str(data.get("status", "asserted"))
        family = "ce"
    elif event_type == "com.chelated.stream.suppression":
        kind = "stream_suppression"
        status = "suppressed"
        family = "stream"
    elif event_type == "com.chelated.prov.derivation":
        kind = "prov_note"
        status = "derived"
        family = "prov"
    else:
        kind = "cloudevent"
        status = str(data.get("status", "asserted"))
        family = "ce"
    claim = _new_claim(
        event_id,
        kind=kind,
        status=status,
        authority=chosen,
        valid_from=valid_from,
        base=True,
        family=family,
        payload=_stored_payload(data),
        audit_only=audit_only,
        reserved_disputed=reserved,
    )
    return [claim], edges, family


def _entity_dicts(raw: object) -> list[dict[str, object]]:
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    raise EnvelopeError("prov_entity", "PROV entity must be an object or a list of objects")


def _derivation_dicts(raw: object) -> list[dict[str, object]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    raise EnvelopeError("wasDerivedFrom", "wasDerivedFrom must be an object or a list")


def _parse_prov(
    envelope: dict[str, object],
    *,
    authority: int,
    tx_hint: int,
) -> tuple[list[Claim], list[Edge], str]:
    nested = envelope.get("prov") if isinstance(envelope.get("prov"), dict) else {}
    assert isinstance(nested, dict)
    raw_entities = envelope.get("prov:entity", envelope.get("entity", nested.get("entity", nested.get("prov:entity"))))
    raw_edges = envelope.get(
        "wasDerivedFrom",
        envelope.get("prov:wasDerivedFrom", nested.get("wasDerivedFrom", nested.get("prov:wasDerivedFrom"))),
    )
    if raw_entities is None and raw_edges is None:
        raise EnvelopeError("prov", "PROV envelope requires an entity or wasDerivedFrom")
    entities = _entity_dicts(raw_entities) if raw_entities is not None else []
    derivations = _derivation_dicts(raw_edges)
    chosen = _clamp_authority(envelope.get("authority"), authority)
    valid_from = _parse_time(envelope.get("time", envelope.get("valid_time")), tx_hint)
    derived_ids = set()
    edges: list[Edge] = []
    for item in derivations:
        generated = item.get("generatedEntity", item.get("prov:generatedEntity", item.get("generated")))
        used = item.get("usedEntity", item.get("prov:usedEntity", item.get("used")))
        if not isinstance(generated, str) or not isinstance(used, str):
            raise EnvelopeError("wasDerivedFrom", "generated and used entity ids are required")
        derived_ids.add(generated)
        edges.append(_new_edge(used, generated, "supports", chosen, True, valid_from))
    claims: list[Claim] = []
    seen: set[str] = set()
    for index, entity in enumerate(entities):
        entity_id = entity.get("id", entity.get("prov:id"))
        if not isinstance(entity_id, str) or not entity_id:
            raise EnvelopeError("prov_entity", "PROV entity id is required")
        if entity_id in seen:
            raise EnvelopeError("prov_entity", f"duplicate entity {entity_id}")
        seen.add(entity_id)
        if "base" in entity:
            base = bool(entity["base"])
        else:
            base = entity_id not in derived_ids
        claims.append(
            _new_claim(
                entity_id,
                kind="prov_entity",
                status="derived" if entity_id in derived_ids else "asserted",
                authority=chosen,
                valid_from=valid_from,
                base=base,
                family="prov",
                payload=_stored_payload(entity),
            )
        )
        if index > 10_000:
            raise EnvelopeError("prov_entity", "too many entities")
    return claims, edges, "prov"


def _status_name(value: object, *, default: str) -> str:
    text = default if value is None else str(value).strip().lower().replace("-", "_")
    if text in {"not_affected", "unaffected", "known_not_affected"}:
        return "not_affected"
    if text in {"affected", "known_affected"}:
        return "affected"
    raise EnvelopeError("vuln_status", f"unsupported vulnerability status {value!r}")


def _parse_osv(
    envelope: dict[str, object],
    *,
    authority: int,
    tx_hint: int,
) -> tuple[list[Claim], list[Edge], str]:
    vuln_id = envelope.get("id")
    if not isinstance(vuln_id, str) or not vuln_id:
        raise EnvelopeError("osv", "OSV id is required")
    if "affected" not in envelope:
        raise EnvelopeError("osv", "OSV affected list is required")
    chosen = _clamp_authority(envelope.get("authority"), authority)
    valid_from = _parse_time(envelope.get("time", envelope.get("valid_time")), tx_hint)
    status = _status_name(envelope.get("status"), default="affected" if envelope.get("affected") else "not_affected")
    claim = _new_claim(
        vuln_id,
        kind="osv",
        status=status,
        authority=chosen,
        valid_from=valid_from,
        base=True,
        family="osv",
        payload={"id": vuln_id, "status": status},
        echo_key=f"{vuln_id}|{status}",
        conflict_key=vuln_id,
    )
    return [claim], [], "osv"


def _parse_csaf(
    envelope: dict[str, object],
    *,
    authority: int,
    tx_hint: int,
) -> tuple[list[Claim], list[Edge], str]:
    chosen = _clamp_authority(envelope.get("authority"), authority)
    valid_from = _parse_time(envelope.get("time", envelope.get("valid_time")), tx_hint)
    flat = envelope.get("csaf_vex")
    vulnerabilities: list[dict[str, object]] = []
    if isinstance(flat, dict):
        vulnerabilities = [flat]
    else:
        raw = envelope.get("vulnerabilities")
        if not isinstance(raw, list):
            raise EnvelopeError("csaf", "CSAF vulnerabilities list is required")
        vulnerabilities = [item for item in raw if isinstance(item, dict)]
    claims: list[Claim] = []
    edges: list[Edge] = []
    for index, vuln in enumerate(vulnerabilities):
        vuln_id = vuln.get("cve", vuln.get("id"))
        if not isinstance(vuln_id, str) or not vuln_id:
            raise EnvelopeError("csaf", "vulnerability id is required")
        product_status = vuln.get("product_status")
        statuses: list[str] = []
        if isinstance(product_status, dict):
            if product_status.get("known_affected") or product_status.get("affected"):
                statuses.append("affected")
            if product_status.get("known_not_affected") or product_status.get("not_affected"):
                statuses.append("not_affected")
        if not statuses:
            statuses.append(_status_name(vuln.get("status"), default="affected"))
        made: list[str] = []
        for status in statuses:
            claim_id = vuln_id if len(statuses) == 1 else f"{vuln_id}:{status}"
            claims.append(
                _new_claim(
                    claim_id,
                    kind="csaf_vex",
                    status=status,
                    authority=chosen,
                    valid_from=valid_from,
                    base=True,
                    family="csaf",
                    payload={"id": vuln_id, "status": status},
                    echo_key=f"{vuln_id}|{status}",
                    conflict_key=vuln_id,
                )
            )
            made.append(claim_id)
        if len(made) == 2:
            edges.append(_new_edge(made[0], made[1], "conflicts", chosen, True, valid_from))
            edges.append(_new_edge(made[1], made[0], "conflicts", chosen, True, valid_from))
        if index > 10_000:
            raise EnvelopeError("csaf", "too many vulnerabilities")
    if not claims:
        raise EnvelopeError("csaf", "CSAF envelope did not contain a vulnerability")
    return claims, edges, "csaf"


def _parse_git(
    envelope: dict[str, object],
    *,
    authority: int,
    tx_hint: int,
) -> tuple[list[Claim], list[Edge], str]:
    chosen = _clamp_authority(envelope.get("authority"), authority)
    valid_from = _parse_time(envelope.get("time", envelope.get("valid_time")), tx_hint)
    claims: list[Claim] = []
    edges: list[Edge] = []
    commit = envelope.get("commit")
    commit_id: str | None = None
    if commit is not None:
        if not isinstance(commit, str) or not (7 <= len(commit) <= 64):
            raise EnvelopeError("git_commit", "commit must be 7 to 64 hex characters")
        if any(char not in "0123456789abcdefABCDEF" for char in commit):
            raise EnvelopeError("git_commit", "commit must be hex")
        commit_id = f"git:{commit.lower()}"
        claims.append(
            _new_claim(
                commit_id,
                kind="git_commit",
                status="committed",
                authority=chosen,
                valid_from=valid_from,
                base=True,
                family="git",
                payload={"commit": commit.lower()},
                echo_key=commit_id,
            )
        )
    symbols = envelope.get("symbols", [])
    if symbols is None:
        symbols = []
    if not isinstance(symbols, list):
        raise EnvelopeError("git_symbol", "symbols must be a list")
    if commit is None and not symbols:
        raise EnvelopeError("git", "git envelope requires a commit or symbols")
    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol:
            raise EnvelopeError("git_symbol", "each symbol must be a non-empty string")
        symbol_id = f"sym:{symbol}"
        claims.append(
            _new_claim(
                symbol_id,
                kind="git_symbol",
                status="defined",
                authority=chosen,
                valid_from=valid_from,
                base=commit_id is None,
                family="git",
                payload={"symbol": symbol},
                echo_key=symbol_id,
            )
        )
        if commit_id is not None:
            edges.append(_new_edge(commit_id, symbol_id, "supports", chosen, True, valid_from))
    return claims, edges, "git"


def _parse_cdc(
    envelope: dict[str, object],
    *,
    authority: int,
    tx_hint: int,
) -> tuple[list[Claim], list[Edge], str]:
    op = str(envelope.get("op", "")).lower()
    if op not in {"c", "u", "d"}:
        raise EnvelopeError("cdc", "op must be c, u, or d")
    table = envelope.get("table")
    pk = envelope.get("pk", envelope.get("primary_key"))
    if not isinstance(table, str) or not table:
        raise EnvelopeError("cdc", "table is required")
    if pk is None or isinstance(pk, bool):
        raise EnvelopeError("cdc", "pk is required")
    pk_text = str(pk)
    chosen = _clamp_authority(envelope.get("authority"), authority)
    valid_from = _parse_time(envelope.get("time", envelope.get("valid_time")), tx_hint)
    if op == "d":
        payload = _mapping(envelope.get("before"), "before")
        status = "erased"
    else:
        payload = _mapping(envelope.get("after"), "after")
        status = "inserted" if op == "c" else "updated"
    claim = _new_claim(
        f"cdc:{table}:{pk_text}:{op}",
        kind="cdc_row",
        status=status,
        authority=chosen,
        valid_from=valid_from,
        base=True,
        family="cdc",
        payload=dict(payload),
        op=op,
        pk=pk_text,
        table=table,
    )
    return [claim], [], "cdc"


def _looks_like_prov(envelope: dict[str, object]) -> bool:
    if "prov:entity" in envelope or "wasDerivedFrom" in envelope or "prov:wasDerivedFrom" in envelope:
        return True
    nested = envelope.get("prov")
    if isinstance(nested, dict) and ("entity" in nested or "wasDerivedFrom" in nested):
        return True
    context = str(envelope.get("@context", ""))
    return "prov" in context.lower() and "entity" in envelope


def _looks_like_csaf(envelope: dict[str, object]) -> bool:
    if isinstance(envelope.get("csaf_vex"), dict):
        return True
    return isinstance(envelope.get("document"), dict) and isinstance(envelope.get("vulnerabilities"), list)


def _looks_like_osv(envelope: dict[str, object]) -> bool:
    return isinstance(envelope.get("id"), str) and "affected" in envelope


def _looks_like_git(envelope: dict[str, object]) -> bool:
    return "commit" in envelope or "symbols" in envelope


def _looks_like_cdc(envelope: dict[str, object]) -> bool:
    op = envelope.get("op")
    return op in {"c", "u", "d", "C", "U", "D"} and "table" in envelope and ("pk" in envelope or "primary_key" in envelope)


def _parse_envelope(
    envelope: dict[str, object],
    *,
    authority: int,
    tx_hint: int,
) -> tuple[list[Claim], list[Edge], str]:
    if envelope.get("specversion") == "1.0":
        return _parse_cloudevent(envelope, authority=authority, tx_hint=tx_hint)
    if _looks_like_prov(envelope):
        return _parse_prov(envelope, authority=authority, tx_hint=tx_hint)
    if _looks_like_csaf(envelope):
        return _parse_csaf(envelope, authority=authority, tx_hint=tx_hint)
    if _looks_like_osv(envelope):
        return _parse_osv(envelope, authority=authority, tx_hint=tx_hint)
    if _looks_like_git(envelope):
        return _parse_git(envelope, authority=authority, tx_hint=tx_hint)
    if _looks_like_cdc(envelope):
        return _parse_cdc(envelope, authority=authority, tx_hint=tx_hint)
    raise EnvelopeError("envelope", "envelope is not CloudEvents, PROV, OSV, CSAF VEX, Git, or CDC")
