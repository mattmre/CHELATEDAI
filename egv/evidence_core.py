"""Stable facade for the Slice 2 evidence-core public API."""

from .canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    canonical_json_bytes,
    canonical_jsonl,
    collection_digest,
    content_id,
    digest_for,
    failure_family_root,
    sha256_digest,
)
from .ledger import EvidenceLedger, SingleWriterLedger
from .projection import InMemoryProjection, QdrantProjection, create_projection, isolated_collection_name, qdrant_available
from .public import PublicCryptographicVerifier, PublicEventChain, PublicProjection
from .receipts import ReceiptJournal, ReceiptSigner, receipt_hash, verify_receipt
from .pilot import run_two_process_smoke

__all__ = [
    "EvidenceLedger",
    "GENESIS_HASH",
    "InMemoryProjection",
    "PublicCryptographicVerifier",
    "PublicEventChain",
    "PublicProjection",
    "QdrantProjection",
    "ReceiptJournal",
    "ReceiptSigner",
    "SingleWriterLedger",
    "canonical_bytes",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_jsonl",
    "collection_digest",
    "content_id",
    "create_projection",
    "digest_for",
    "failure_family_root",
    "isolated_collection_name",
    "qdrant_available",
    "receipt_hash",
    "run_two_process_smoke",
    "sha256_digest",
    "verify_receipt",
]
