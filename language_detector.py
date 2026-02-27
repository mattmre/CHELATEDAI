"""
Language Detection Module for ChelatedAI

Provides lightweight language detection with optional langdetect backend
and graceful fallback to character-range heuristics.
"""

from typing import Dict, List, Tuple
from chelation_logger import get_logger

# Try importing langdetect; graceful fallback if unavailable
try:
    import langdetect as _langdetect
    _HAS_LANGDETECT = True
except ImportError:
    _langdetect = None
    _HAS_LANGDETECT = False


# Unicode character ranges for heuristic detection
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
]

_HIRAGANA_RANGE = (0x3040, 0x309F)
_KATAKANA_RANGE = (0x30A0, 0x30FF)

_HANGUL_RANGES = [
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
]

_CYRILLIC_RANGE = (0x0400, 0x04FF)

_ARABIC_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
]

_DEVANAGARI_RANGE = (0x0900, 0x097F)

_THAI_RANGE = (0x0E00, 0x0E7F)


def _in_ranges(cp: int, ranges) -> bool:
    """Check if codepoint is within any of the given ranges."""
    if isinstance(ranges, tuple) and len(ranges) == 2 and isinstance(ranges[0], int):
        return ranges[0] <= cp <= ranges[1]
    for start, end in ranges:
        if start <= cp <= end:
            return True
    return False


def _is_latin(cp: int) -> bool:
    """Check if codepoint is Latin script."""
    return (
        (0x0041 <= cp <= 0x005A)   # A-Z
        or (0x0061 <= cp <= 0x007A)  # a-z
        or (0x00C0 <= cp <= 0x024F)  # Latin Extended
    )


class LanguageDetector:
    """
    Lightweight language detector with optional langdetect backend.

    Falls back to character-range heuristics when langdetect is unavailable
    or for very short texts where langdetect is unreliable.
    """

    # Minimum text length for langdetect to be reliable
    MIN_LANGDETECT_LENGTH = 20

    # Default confidence threshold
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, confidence_threshold: float = None,
                 default_language: str = "en",
                 use_cache: bool = True,
                 cache_max_size: int = 10000):
        """
        Initialize language detector.

        Args:
            confidence_threshold: Minimum confidence to trust langdetect result.
                Falls back to heuristics below this threshold.
            default_language: Default language code when detection is uncertain.
            use_cache: Whether to cache detection results for repeated texts.
            cache_max_size: Maximum number of cached entries.
        """
        self.logger = get_logger()
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None
            else self.DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.default_language = default_language
        self.use_cache = use_cache
        self.cache_max_size = cache_max_size
        self._cache: Dict[str, str] = {}
        self._has_langdetect = _HAS_LANGDETECT

        self.logger.log_event(
            "language_detector_init",
            f"LanguageDetector initialized (langdetect={'available' if self._has_langdetect else 'unavailable'})",
            backend="langdetect" if self._has_langdetect else "heuristic",
            default_language=default_language,
        )

    @property
    def backend(self) -> str:
        """Return which detection backend is active."""
        return "langdetect" if self._has_langdetect else "heuristic"

    def detect(self, text: str) -> str:
        """
        Detect language of a single text.

        Args:
            text: Input text string.

        Returns:
            ISO 639-1 language code (e.g. 'en', 'de', 'zh', 'ja', 'ko').
        """
        if not text or not text.strip():
            return self.default_language

        # Check cache
        if self.use_cache and text in self._cache:
            return self._cache[text]

        result = self._detect_single(text)

        # Cache result
        if self.use_cache:
            if len(self._cache) >= self.cache_max_size:
                # Simple eviction: clear half the cache
                keys = list(self._cache.keys())
                for k in keys[:len(keys) // 2]:
                    del self._cache[k]
            self._cache[text] = result

        return result

    def detect_batch(self, texts: List[str]) -> List[str]:
        """
        Detect languages for a batch of texts.

        Args:
            texts: List of input text strings.

        Returns:
            List of ISO 639-1 language codes, one per input text.
        """
        return [self.detect(text) for text in texts]

    def detect_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Detect language with confidence score.

        Args:
            text: Input text string.

        Returns:
            Tuple of (language_code, confidence) where confidence is 0.0-1.0.
        """
        if not text or not text.strip():
            return self.default_language, 0.0

        # Try langdetect first for longer texts
        if (self._has_langdetect
                and len(text.strip()) >= self.MIN_LANGDETECT_LENGTH):
            try:
                results = _langdetect.detect_langs(text)
                if results:
                    best = results[0]
                    return str(best.lang), float(best.prob)
            except Exception:
                pass

        # Fall back to heuristics
        lang, confidence = self._heuristic_detect(text)
        return lang, confidence

    def _detect_single(self, text: str) -> str:
        """Internal detection for a single text."""
        stripped = text.strip()

        # For short texts, use heuristics (langdetect is unreliable)
        if len(stripped) < self.MIN_LANGDETECT_LENGTH:
            lang, confidence = self._heuristic_detect(stripped)
            if confidence >= self.confidence_threshold:
                return lang
            return self.default_language

        # Try langdetect if available
        if self._has_langdetect:
            try:
                results = _langdetect.detect_langs(stripped)
                if results:
                    best = results[0]
                    if float(best.prob) >= self.confidence_threshold:
                        return str(best.lang)
            except Exception as e:
                self.logger.log_event(
                    "langdetect_error",
                    f"langdetect failed, falling back to heuristics: {e}",
                    level="WARNING",
                )

        # Fall back to heuristics
        lang, confidence = self._heuristic_detect(stripped)
        if confidence >= self.confidence_threshold:
            return lang

        return self.default_language

    def _heuristic_detect(self, text: str) -> Tuple[str, float]:
        """
        Character-range heuristic detection.

        Returns:
            Tuple of (language_code, confidence).
        """
        if not text:
            return self.default_language, 0.0

        # Count characters by script
        counts = {
            "cjk": 0,
            "hiragana": 0,
            "katakana": 0,
            "hangul": 0,
            "cyrillic": 0,
            "arabic": 0,
            "devanagari": 0,
            "thai": 0,
            "latin": 0,
        }
        total_alpha = 0

        for char in text:
            cp = ord(char)
            if _in_ranges(cp, _CJK_RANGES):
                counts["cjk"] += 1
                total_alpha += 1
            elif _in_ranges(cp, _HIRAGANA_RANGE):
                counts["hiragana"] += 1
                total_alpha += 1
            elif _in_ranges(cp, _KATAKANA_RANGE):
                counts["katakana"] += 1
                total_alpha += 1
            elif _in_ranges(cp, _HANGUL_RANGES):
                counts["hangul"] += 1
                total_alpha += 1
            elif _in_ranges(cp, _CYRILLIC_RANGE):
                counts["cyrillic"] += 1
                total_alpha += 1
            elif _in_ranges(cp, _ARABIC_RANGES):
                counts["arabic"] += 1
                total_alpha += 1
            elif _in_ranges(cp, _DEVANAGARI_RANGE):
                counts["devanagari"] += 1
                total_alpha += 1
            elif _in_ranges(cp, _THAI_RANGE):
                counts["thai"] += 1
                total_alpha += 1
            elif _is_latin(cp):
                counts["latin"] += 1
                total_alpha += 1

        if total_alpha == 0:
            return self.default_language, 0.0

        # Determine dominant script
        # Japanese uses CJK + hiragana/katakana
        japanese_chars = counts["hiragana"] + counts["katakana"]
        if japanese_chars > 0 and (japanese_chars + counts["cjk"]) / total_alpha > 0.3:
            confidence = (japanese_chars + counts["cjk"]) / total_alpha
            return "ja", min(confidence, 1.0)

        # Korean uses hangul
        if counts["hangul"] / total_alpha > 0.3:
            confidence = counts["hangul"] / total_alpha
            return "ko", min(confidence, 1.0)

        # Chinese uses CJK without Japanese kana
        if counts["cjk"] / total_alpha > 0.3 and japanese_chars == 0:
            confidence = counts["cjk"] / total_alpha
            return "zh", min(confidence, 1.0)

        # Cyrillic -> Russian (most common Cyrillic language)
        if counts["cyrillic"] / total_alpha > 0.3:
            confidence = counts["cyrillic"] / total_alpha
            return "ru", min(confidence, 1.0)

        # Arabic
        if counts["arabic"] / total_alpha > 0.3:
            confidence = counts["arabic"] / total_alpha
            return "ar", min(confidence, 1.0)

        # Devanagari -> Hindi
        if counts["devanagari"] / total_alpha > 0.3:
            confidence = counts["devanagari"] / total_alpha
            return "hi", min(confidence, 1.0)

        # Thai
        if counts["thai"] / total_alpha > 0.3:
            confidence = counts["thai"] / total_alpha
            return "th", min(confidence, 1.0)

        # Latin script -> default to English
        if counts["latin"] / total_alpha > 0.5:
            # For Latin, confidence is lower since many languages use Latin
            confidence = counts["latin"] / total_alpha * 0.6
            return "en", min(confidence, 1.0)

        return self.default_language, 0.0

    def clear_cache(self):
        """Clear the detection cache."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Return current cache size."""
        return len(self._cache)
