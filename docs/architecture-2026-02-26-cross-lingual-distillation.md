# Architecture: Cross-Lingual Distillation
Date: 2026-02-26 | Session: 21 | Status: Approved

## Summary
Language-aware teacher routing for cross-lingual embedding alignment.

## New Files
- language_detector.py: Lightweight LanguageDetector (langdetect backend, graceful fallback)
- cross_lingual_distillation.py: CrossLingualTeacherRouter + LanguageTeacherMapping

## Key Design: Duck-Typing
CrossLingualTeacherRouter implements same API as TeacherDistillationHelper/EnsembleTeacherHelper:
- get_teacher_embeddings(texts, target_dim=None)
- generate_distillation_targets(texts, student_embeddings, teacher_weight=None)
- compute_alignment_metric(student_embeds, teacher_embeds=None, texts=None)
Can drop into engine.teacher_helper with zero training loop changes.

## Language Routing Flow
1. Detect language per text (langdetect, fallback for short texts)
2. Group texts by language
3. Route each group to language-specific teacher
4. Project all to target_dim, reassemble in original order

## Config Presets (5 presets)
- en_de, en_zh, en_ja: Language-pair specialists
- multilingual_universal: Single multilingual teacher for all languages
- multilingual_hybrid: Best English model + multilingual fallback

## Recommended Models
- General: paraphrase-multilingual-MiniLM-L12-v2 (384d, 50+ langs)
- English anchor: all-mpnet-base-v2 (768d, best English quality)
- Bridge: paraphrase-multilingual-mpnet-base-v2 (768d, 50+ langs)

## Test Plan: ~40 tests across test_cross_lingual_distillation.py + test_language_detector.py
