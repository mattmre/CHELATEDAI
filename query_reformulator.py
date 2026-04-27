"""Lightweight query reformulation scaffolding for future adaptive RAG loops."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List

from chelation_logger import get_logger


@dataclass
class QueryReformulation:
    """One reformulated query candidate."""

    text: str
    strategy: str


class QueryReformulator:
    """Generate deterministic query variants without external LLM dependencies."""

    def __init__(self, stopwords: Iterable[str] | None = None, logger=None):
        self.stopwords = set(stopwords or {"the", "a", "an", "of", "to", "and", "or", "for", "in"})
        self.logger = logger or get_logger()

    def reformulate(self, query: str, max_variants: int = 3) -> List[QueryReformulation]:
        if max_variants < 1:
            raise ValueError("max_variants must be >= 1")
        normalized = " ".join(re.findall(r"[A-Za-z0-9_+-]+", query.lower()))
        tokens = normalized.split()
        if not tokens:
            raise ValueError("query must contain at least one token")

        variants = [QueryReformulation(text=query.strip(), strategy="original")]
        keyword_tokens = [token for token in tokens if token not in self.stopwords]
        if keyword_tokens:
            variants.append(QueryReformulation(text=" ".join(keyword_tokens), strategy="stopword_removed"))
        if len(keyword_tokens) > 2:
            variants.append(QueryReformulation(text=" ".join(keyword_tokens[:2]), strategy="focused_prefix"))

        deduped = []
        seen = set()
        for variant in variants:
            key = variant.text.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(variant)
            if len(deduped) >= max_variants:
                break
        self.logger.log_event(
            "query_reformulated",
            "Generated query reformulation variants",
            variant_count=len(deduped),
            strategies=[variant.strategy for variant in deduped],
            level="DEBUG",
        )
        return deduped

