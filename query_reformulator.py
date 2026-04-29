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


def query_lexical_features(query: str) -> dict:
    """Compute lightweight query-shape features usable before retrieval."""

    normalized = " ".join(re.findall(r"[A-Za-z0-9_+-]+", query.lower()))
    tokens = normalized.split()
    stopwords = {"the", "a", "an", "of", "to", "and", "or", "for", "in"}
    negations = {"no", "not", "without", "lack", "lacks", "never"}
    claim_cues = {
        "increase",
        "increased",
        "decrease",
        "decreased",
        "risk",
        "treat",
        "treats",
        "used",
        "associated",
        "affect",
        "facilitates",
    }
    token_count = len(tokens)
    return {
        "token_count": token_count,
        "char_count": len(query),
        "stopword_ratio": (sum(token in stopwords for token in tokens) / token_count) if token_count else 0.0,
        "numeric_token_count": sum(any(char.isdigit() for char in token) for token in tokens),
        "negation_count": sum(token in negations for token in tokens),
        "claim_cue_count": sum(token in claim_cues for token in tokens),
    }


def should_apply_reformulation(query: str, policy: str = "always") -> bool:
    """Return whether query reformulation should run for a query under a policy."""

    features = query_lexical_features(query)
    token_count = int(features["token_count"])
    stopword_ratio = float(features["stopword_ratio"])
    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy == "selective_low_specificity":
        if token_count == 0:
            return False
        return token_count <= 6 or stopword_ratio >= 0.35
    if policy == "selective_high_specificity":
        return token_count >= 8 and stopword_ratio <= 0.25
    if policy == "selective_claim_cue":
        return bool(features["claim_cue_count"] or features["negation_count"] or features["numeric_token_count"])
    raise ValueError(f"unsupported query reformulation policy: {policy}")

