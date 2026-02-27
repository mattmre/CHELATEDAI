"""
Unit Tests for Language Detector Module

Tests the LanguageDetector class with heuristic fallback detection.
All tests run without langdetect installed.
"""

import unittest
from unittest.mock import MagicMock, patch

from language_detector import LanguageDetector


class TestLanguageDetectorHeuristics(unittest.TestCase):
    """Test heuristic-based language detection (no langdetect dependency)."""

    def setUp(self):
        self.logger_patcher = patch("language_detector.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()
        # Force heuristic mode by disabling langdetect
        self.detector = LanguageDetector(default_language="en")
        self.detector._has_langdetect = False

    def tearDown(self):
        self.logger_patcher.stop()

    def test_detect_english(self):
        """Test detection of English (Latin script) text."""
        result = self.detector.detect("This is an English sentence for testing")
        self.assertEqual(result, "en")

    def test_detect_chinese(self):
        """Test detection of Chinese (CJK) text."""
        result = self.detector.detect("这是一个中文句子用于测试语言检测")
        self.assertEqual(result, "zh")

    def test_detect_japanese(self):
        """Test detection of Japanese text (hiragana + kanji)."""
        result = self.detector.detect("これは日本語のテスト文です")
        self.assertEqual(result, "ja")

    def test_detect_korean(self):
        """Test detection of Korean (Hangul) text."""
        result = self.detector.detect("이것은 한국어 테스트 문장입니다")
        self.assertEqual(result, "ko")

    def test_detect_russian(self):
        """Test detection of Russian (Cyrillic) text."""
        result = self.detector.detect("Это русское предложение для тестирования")
        self.assertEqual(result, "ru")

    def test_detect_arabic(self):
        """Test detection of Arabic text."""
        result = self.detector.detect("هذه جملة عربية للاختبار")
        self.assertEqual(result, "ar")

    def test_detect_hindi(self):
        """Test detection of Hindi (Devanagari) text."""
        result = self.detector.detect("यह एक हिंदी वाक्य है परीक्षण के लिए")
        self.assertEqual(result, "hi")

    def test_detect_thai(self):
        """Test detection of Thai text."""
        result = self.detector.detect("นี่คือประโยคภาษาไทยสำหรับการทดสอบ")
        self.assertEqual(result, "th")

    def test_detect_empty_string(self):
        """Test detection of empty string returns default language."""
        result = self.detector.detect("")
        self.assertEqual(result, "en")

    def test_detect_whitespace_only(self):
        """Test detection of whitespace-only string returns default."""
        result = self.detector.detect("   \t\n  ")
        self.assertEqual(result, "en")

    def test_detect_short_text_below_threshold(self):
        """Test that very short text uses heuristics even if langdetect available."""
        result = self.detector.detect("Hi")
        self.assertEqual(result, "en")

    def test_detect_batch(self):
        """Test batch detection returns correct languages."""
        texts = [
            "Hello world",
            "这是中文",
            "これはテスト",
        ]
        results = self.detector.detect_batch(texts)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "en")
        self.assertEqual(results[1], "zh")
        self.assertEqual(results[2], "ja")

    def test_detect_batch_empty(self):
        """Test batch detection with empty list."""
        results = self.detector.detect_batch([])
        self.assertEqual(results, [])


class TestLanguageDetectorCache(unittest.TestCase):
    """Test caching behavior of LanguageDetector."""

    def setUp(self):
        self.logger_patcher = patch("language_detector.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.logger_patcher.stop()

    def test_cache_stores_results(self):
        """Test that detection results are cached."""
        detector = LanguageDetector(use_cache=True)
        detector._has_langdetect = False
        text = "This is a test sentence for caching"
        detector.detect(text)
        self.assertEqual(detector.cache_size(), 1)
        # Second call should hit cache
        result = detector.detect(text)
        self.assertEqual(result, "en")
        self.assertEqual(detector.cache_size(), 1)

    def test_cache_disabled(self):
        """Test that cache can be disabled."""
        detector = LanguageDetector(use_cache=False)
        detector._has_langdetect = False
        detector.detect("This is a test sentence for caching")
        self.assertEqual(detector.cache_size(), 0)

    def test_cache_eviction(self):
        """Test cache eviction when max size is reached."""
        detector = LanguageDetector(use_cache=True, cache_max_size=5)
        detector._has_langdetect = False
        for i in range(6):
            detector.detect(f"Test sentence number {i} for eviction testing")
        # After eviction, cache should have fewer entries
        self.assertLessEqual(detector.cache_size(), 5)

    def test_clear_cache(self):
        """Test clearing the cache."""
        detector = LanguageDetector(use_cache=True)
        detector._has_langdetect = False
        detector.detect("Hello world testing cache clear")
        self.assertGreater(detector.cache_size(), 0)
        detector.clear_cache()
        self.assertEqual(detector.cache_size(), 0)


class TestLanguageDetectorConfidence(unittest.TestCase):
    """Test confidence-based detection."""

    def setUp(self):
        self.logger_patcher = patch("language_detector.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.logger_patcher.stop()

    def test_detect_with_confidence_returns_tuple(self):
        """Test detect_with_confidence returns (lang, confidence)."""
        detector = LanguageDetector()
        detector._has_langdetect = False
        lang, conf = detector.detect_with_confidence("This is an English test sentence")
        self.assertIsInstance(lang, str)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_detect_with_confidence_empty(self):
        """Test detect_with_confidence with empty string."""
        detector = LanguageDetector()
        detector._has_langdetect = False
        lang, conf = detector.detect_with_confidence("")
        self.assertEqual(lang, "en")
        self.assertEqual(conf, 0.0)

    def test_confidence_threshold_affects_result(self):
        """Test that high confidence threshold falls back to default for ambiguous text."""
        detector = LanguageDetector(confidence_threshold=0.99, default_language="en")
        detector._has_langdetect = False
        # Numerics/symbols have zero confidence -- should fall back to default
        result = detector.detect("12345!@#$%")
        self.assertEqual(result, "en")


class TestLanguageDetectorBackend(unittest.TestCase):
    """Test backend property and initialization."""

    def setUp(self):
        self.logger_patcher = patch("language_detector.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.logger_patcher.stop()

    def test_backend_property_heuristic(self):
        """Test backend property when langdetect is not available."""
        detector = LanguageDetector()
        detector._has_langdetect = False
        self.assertEqual(detector.backend, "heuristic")

    def test_backend_property_langdetect(self):
        """Test backend property when langdetect is available."""
        detector = LanguageDetector()
        detector._has_langdetect = True
        self.assertEqual(detector.backend, "langdetect")

    def test_default_language_configurable(self):
        """Test that default language is configurable."""
        detector = LanguageDetector(default_language="de")
        detector._has_langdetect = False
        result = detector.detect("")
        self.assertEqual(result, "de")


if __name__ == "__main__":
    unittest.main(verbosity=2)
