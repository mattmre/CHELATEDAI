"""
Cross-Lingual Distillation Module for ChelatedAI

Provides language-aware teacher routing for cross-lingual embedding alignment.
CrossLingualTeacherRouter duck-types TeacherDistillationHelper / EnsembleTeacherHelper
and can drop into engine.teacher_helper with zero training loop changes.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from chelation_logger import get_logger
from config import ChelationConfig
from language_detector import LanguageDetector
from teacher_distillation import (
    DimensionProjection,
    TeacherDistillationHelper,
)


class LanguageTeacherMapping:
    """
    Maps language codes to teacher model names.

    Supports exact language matches and a fallback/default teacher.
    """

    def __init__(self, mappings: Optional[Dict[str, str]] = None,
                 default_teacher: str = None):
        """
        Args:
            mappings: Dict mapping ISO 639-1 language codes to model names.
                Example: {"en": "all-mpnet-base-v2", "de": "paraphrase-multilingual-MiniLM-L12-v2"}
            default_teacher: Fallback model for unmapped languages.
        """
        self.mappings = dict(mappings) if mappings else {}
        self.default_teacher = (
            default_teacher or ChelationConfig.DEFAULT_TEACHER_MODEL
        )

    def get_teacher_for_language(self, lang: str) -> str:
        """
        Get teacher model name for a language code.

        Args:
            lang: ISO 639-1 language code (e.g. 'en', 'de', 'zh').

        Returns:
            Teacher model name string.
        """
        return self.mappings.get(lang, self.default_teacher)

    def get_unique_teachers(self) -> List[str]:
        """Return list of unique teacher model names referenced by this mapping."""
        teachers = set(self.mappings.values())
        teachers.add(self.default_teacher)
        return sorted(teachers)

    def has_language(self, lang: str) -> bool:
        """Check if a specific language mapping exists."""
        return lang in self.mappings

    def add_mapping(self, lang: str, teacher_model: str):
        """Add or update a language-to-teacher mapping."""
        self.mappings[lang] = teacher_model

    def __repr__(self):
        return (
            f"LanguageTeacherMapping(mappings={self.mappings}, "
            f"default_teacher='{self.default_teacher}')"
        )


class CrossLingualTeacherRouter:
    """
    Language-aware teacher router for cross-lingual distillation.

    Duck-types the same API as TeacherDistillationHelper and EnsembleTeacherHelper:
    - get_teacher_embeddings(texts, target_dim=None)
    - generate_distillation_targets(texts, student_embeddings, teacher_weight=None)
    - compute_alignment_metric(student_embeds, teacher_embeds=None, texts=None)

    Routing flow:
    1. Detect language per text
    2. Group texts by language
    3. Route each group to language-specific teacher
    4. Project all to target_dim, reassemble in original order
    """

    def __init__(self, language_mapping: LanguageTeacherMapping,
                 detector: Optional[LanguageDetector] = None,
                 projection_enabled: bool = True,
                 projection_hidden_dim: Optional[int] = None,
                 teacher_weight: float = 0.5):
        """
        Args:
            language_mapping: LanguageTeacherMapping instance defining lang->teacher routes.
            detector: Optional LanguageDetector instance. Creates default if None.
            projection_enabled: Whether to auto-project when teacher dims differ from target.
            projection_hidden_dim: Optional bottleneck for projection layers.
            teacher_weight: Default teacher weight for distillation targets.
        """
        self.logger = get_logger()
        self.language_mapping = language_mapping
        self.detector = detector or LanguageDetector()
        self._projection_enabled = projection_enabled
        self._projection_hidden_dim = projection_hidden_dim
        self.teacher_weight = teacher_weight

        # Lazy-loaded teacher helpers: model_name -> TeacherDistillationHelper
        self._teachers: Dict[str, TeacherDistillationHelper] = {}
        # Projection layers: model_name -> DimensionProjection
        self._projections: Dict[str, DimensionProjection] = {}
        self._xl_records = []
        self._xl_batches = []
        self._xl_student = None
        self._live_alpha = None
        self._live_parts = []
        self._record_live = False
        self._accumulate_live = False
        self.logger.log_event(
            "cross_lingual_router_init",
            "CrossLingualTeacherRouter initialized",
            num_language_mappings=len(language_mapping.mappings),
            default_teacher=language_mapping.default_teacher,
            projection_enabled=projection_enabled,
        )

    def check_dimension_compatibility(self, student_dim: int) -> bool:
        """Projection-enabled routing stays compatible. Unknown dims do not raise.

        ``run_offline_distillation`` reads ``teacher_dim`` only after False.
        Returning True while projection is enabled skips that read and
        continues into ``generate_distillation_targets``.
        """
        if self._projection_enabled:
            return True
        mismatched = None
        for teacher in self._teachers.values():
            dim = getattr(teacher, "teacher_dim", None)
            if isinstance(dim, int) and dim != int(student_dim):
                mismatched = dim
                break
        if mismatched is None:
            return True
        self.teacher_dim = mismatched
        return False

    def begin_live_distillation(self):
        """Drop projection graphs retained by an earlier routed target build."""
        self._xl_records = []
        self._xl_batches = []
        self._xl_student = None
        self._live_alpha = None
        self._live_parts = []
        self._accumulate_live = True

    def _blend_xl_records(self, records, student, alpha):
        """Scatter one batch through the current maps. Indices are batch-local."""
        acc = torch.zeros_like(student)
        for indices, kind, key, tensor in records:
            if kind == "proj":
                projected = self._projections[key].project_tensor(tensor)
            else:
                projected = tensor
            index = torch.tensor(indices, dtype=torch.long)
            acc[index] = projected
        blended = alpha * acc + (1.0 - alpha) * student
        norms = torch.linalg.vector_norm(blended, dim=1, keepdim=True).clamp_min(1e-9)
        return blended / norms

    def _store_xl_batch(self, record):
        """Keep one routed batch. A standalone generate replaces earlier batches."""
        if not self._accumulate_live:
            self._xl_batches = []
            self._live_parts = []
        self._xl_batches.append(record)
        if record[0] == "const":
            self._live_parts.append(record[1])
            return
        _kind, records, student, alpha = record
        self._live_parts.append(self._blend_xl_records(records, student, alpha))

    def take_live_distillation_targets(self):
        parts = self._live_parts
        self._live_parts = []
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def recompute_live_targets(self):
        """Rebuild each routed batch, then concatenate.

        Record indices are local to the batch that produced them. Scattering
        every batch into the last student tensor keeps only that batch's shape.
        """
        if not self._xl_batches:
            return None
        parts = []
        for record in self._xl_batches:
            if record[0] == "const":
                parts.append(record[1])
                continue
            _kind, records, student, alpha = record
            parts.append(self._blend_xl_records(records, student, alpha))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def _get_or_create_teacher(self, model_name: str) -> TeacherDistillationHelper:
        """Get or lazily create a teacher helper for the given model name."""
        if model_name not in self._teachers:
            self._teachers[model_name] = TeacherDistillationHelper(
                teacher_model_name=model_name,
                projection_enabled=self._projection_enabled,
                projection_hidden_dim=self._projection_hidden_dim,
            )
            self.logger.log_event(
                "teacher_created",
                f"Created teacher helper for model: {model_name}",
                model_name=model_name,
            )
        return self._teachers[model_name]

    def _group_by_language(self, texts: List[str]) -> Dict[str, List[Tuple[int, str]]]:
        """
        Group texts by detected language.

        Returns:
            Dict mapping language code to list of (original_index, text) tuples.
        """
        languages = self.detector.detect_batch(texts)
        groups: Dict[str, List[Tuple[int, str]]] = {}
        for idx, (text, lang) in enumerate(zip(texts, languages)):
            if lang not in groups:
                groups[lang] = []
            groups[lang].append((idx, text))
        return groups

    def _ensure_projection(self, model_name: str, teacher_dim: int,
                           target_dim: int) -> Optional[DimensionProjection]:
        """Create or retrieve projection for a teacher model if dims differ."""
        if teacher_dim == target_dim:
            return None
        key = f"{model_name}_{teacher_dim}_{target_dim}"
        if key not in self._projections:
            self._projections[key] = DimensionProjection(
                teacher_dim, target_dim,
                hidden_dim=self._projection_hidden_dim,
            )
            self.logger.log_event(
                "projection_created",
                f"Created projection: {teacher_dim} -> {target_dim} for {model_name}",
                model_name=model_name,
                teacher_dim=teacher_dim,
                target_dim=target_dim,
            )
        return self._projections[key]

    def get_teacher_embeddings(self, texts: List[str],
                               target_dim: Optional[int] = None) -> np.ndarray:
        """
        Get teacher embeddings with language-aware routing.

        Args:
            texts: List of text strings.
            target_dim: If specified, project all embeddings to this dimension.

        Returns:
            Numpy array of teacher embeddings [batch_size, dim].
        """
        if not texts:
            return np.array([])

        # Group texts by language
        lang_groups = self._group_by_language(texts)

        # Collect embeddings per group
        result = [None] * len(texts)

        for lang, items in lang_groups.items():
            model_name = self.language_mapping.get_teacher_for_language(lang)
            teacher = self._get_or_create_teacher(model_name)

            group_texts = [text for _, text in items]
            group_indices = [idx for idx, _ in items]

            # Get teacher embeddings for this language group
            group_embeddings = teacher.get_teacher_embeddings(group_texts)

            if len(group_embeddings) == 0:
                continue

            # Project to target_dim if needed. A live recording keeps the map
            # in the graph for generate_distillation_targets.
            if (target_dim is not None
                    and group_embeddings.shape[-1] != target_dim
                    and self._projection_enabled):
                proj = self._ensure_projection(
                    model_name, group_embeddings.shape[-1], target_dim,
                )
                if proj is not None:
                    key = f"{model_name}_{group_embeddings.shape[-1]}_{target_dim}"
                    if self._record_live:
                        teacher_tensor = torch.from_numpy(
                            np.ascontiguousarray(group_embeddings, dtype=np.float32)
                        )
                        live = proj.project_tensor(teacher_tensor)
                        self._xl_records.append(
                            (group_indices, "proj", key, teacher_tensor.detach())
                        )
                        group_embeddings = live.detach().cpu().numpy()
                    else:
                        group_embeddings = proj.project_numpy(group_embeddings)
            elif self._record_live:
                raw = torch.from_numpy(
                    np.ascontiguousarray(group_embeddings, dtype=np.float32)
                )
                self._xl_records.append((group_indices, "raw", None, raw.detach()))

            # Place embeddings back in original order
            for i, idx in enumerate(group_indices):
                result[idx] = group_embeddings[i]

        # Assemble final array
        # Determine output dimension
        sample = next((r for r in result if r is not None), None)
        if sample is None:
            return np.array([])

        observed_dims = sorted({int(r.shape[-1]) for r in result if r is not None})
        if len(observed_dims) > 1:
            if target_dim is None:
                raise ValueError(
                    "Inconsistent teacher dimensions found across language routes. "
                    "Provide target_dim to unify embeddings."
                )
            raise ValueError(
                "Inconsistent teacher dimensions remained after routing. "
                "Enable projection to use target_dim across heterogeneous teachers."
            )

        out_dim = observed_dims[0]
        output = np.zeros((len(texts), out_dim), dtype=np.float32)
        for i, emb in enumerate(result):
            if emb is not None:
                output[i] = emb

        return output

    def generate_distillation_targets(self, texts: List[str],
                                      student_embeddings: np.ndarray = None,
                                      teacher_weight: Optional[float] = None,
                                      current_embeddings: np.ndarray = None,
                                      ) -> np.ndarray:
        """
        Generate distillation targets using language-aware teacher routing.

        Args:
            texts: Original text strings.
            student_embeddings: Current student embeddings [N, dim].
            teacher_weight: Weight for teacher guidance (0=student only, 1=teacher only).
                Uses self.teacher_weight if None.

        Returns:
            Target embeddings [N, dim].
        """
        if teacher_weight is None:
            teacher_weight = self.teacher_weight
        if student_embeddings is None:
            student_embeddings = current_embeddings

        if teacher_weight == 0.0:
            if self._accumulate_live and student_embeddings is not None:
                const = torch.from_numpy(
                    np.ascontiguousarray(student_embeddings, dtype=np.float32)
                ).detach()
                self._store_xl_batch(("const", const))
            return student_embeddings.copy()

        start = len(self._xl_records)
        student_dim = student_embeddings.shape[-1]
        self._record_live = True
        try:
            teacher_embeds = self.get_teacher_embeddings(texts, target_dim=student_dim)
        finally:
            self._record_live = False
        new_records = list(self._xl_records[start:])

        if len(teacher_embeds) == 0:
            if self._accumulate_live:
                self._xl_records = self._xl_records[:start]
                const = torch.from_numpy(
                    np.ascontiguousarray(student_embeddings, dtype=np.float32)
                ).detach()
                self._store_xl_batch(("const", const))
            else:
                self._xl_records = []
            return student_embeddings.copy()

        projected = any(kind == "proj" for _, kind, _, _ in new_records)
        if projected:
            # Hinton, Vinyals, and Dean, 2015, arXiv:1503.02531.
            if not self._accumulate_live:
                self._xl_records = new_records
            student_tensor = torch.from_numpy(
                np.ascontiguousarray(student_embeddings, dtype=np.float32)
            ).detach()
            alpha = float(teacher_weight)
            self._xl_student = student_tensor
            self._live_alpha = alpha
            self._store_xl_batch(("live", new_records, student_tensor, alpha))
            return self._live_parts[-1].detach().cpu().numpy()
        if self._accumulate_live:
            self._xl_records = self._xl_records[:start]
        else:
            self._xl_records = []

        # Blend: target = (1 - alpha) * student + alpha * teacher
        alpha = teacher_weight
        blended = (1 - alpha) * student_embeddings + alpha * teacher_embeds

        # Normalize to unit sphere
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        targets = blended / norms

        self.logger.log_event(
            "cross_lingual_targets_generated",
            f"Generated {len(targets)} cross-lingual distillation targets",
            level="DEBUG",
            num_targets=len(targets),
            teacher_weight=teacher_weight,
        )
        if self._accumulate_live:
            const = torch.from_numpy(
                np.ascontiguousarray(targets, dtype=np.float32)
            ).detach()
            self._store_xl_batch(("const", const))

        return targets

    def compute_alignment_metric(self, student_embeds: np.ndarray,
                                 teacher_embeds: Optional[np.ndarray] = None,
                                 texts: Optional[List[str]] = None) -> float:
        """
        Compute alignment metric between student and teacher embeddings.

        If teacher_embeds is None and texts is provided, generates teacher
        embeddings via language-aware routing.

        Args:
            student_embeds: Student embeddings [N, dim].
            teacher_embeds: Optional teacher embeddings [N, dim].
            texts: Optional texts for generating teacher embeddings.

        Returns:
            Average cosine similarity (0-1).
        """
        if len(student_embeds) == 0:
            return 0.0

        # If no teacher_embeds provided, generate them from texts
        if teacher_embeds is None:
            if texts is not None and len(texts) == len(student_embeds):
                student_dim = student_embeds.shape[-1]
                teacher_embeds = self.get_teacher_embeddings(
                    texts, target_dim=student_dim,
                )
            else:
                return 0.0

        if len(teacher_embeds) == 0:
            return 0.0

        # Handle dimension mismatch
        if student_embeds.shape != teacher_embeds.shape:
            if (self._projection_enabled
                    and len(student_embeds.shape) == 2
                    and len(teacher_embeds.shape) == 2
                    and student_embeds.shape[0] == teacher_embeds.shape[0]
                    and student_embeds.shape[1] != teacher_embeds.shape[1]):
                proj = self._ensure_projection(
                    "_alignment",
                    teacher_embeds.shape[1],
                    student_embeds.shape[1],
                )
                if proj is not None:
                    teacher_embeds = proj.project_numpy(teacher_embeds)
                else:
                    return 0.0
            else:
                return 0.0

        # Compute row-wise cosine similarities
        student_norm = student_embeds / (
            np.linalg.norm(student_embeds, axis=1, keepdims=True) + 1e-9
        )
        teacher_norm = teacher_embeds / (
            np.linalg.norm(teacher_embeds, axis=1, keepdims=True) + 1e-9
        )
        similarities = np.sum(student_norm * teacher_norm, axis=1)
        return float(np.mean(similarities))

    def get_language_stats(self, texts: List[str]) -> Dict[str, int]:
        """
        Get language distribution statistics for a batch of texts.

        Args:
            texts: List of text strings.

        Returns:
            Dict mapping language codes to counts.
        """
        languages = self.detector.detect_batch(texts)
        stats: Dict[str, int] = {}
        for lang in languages:
            stats[lang] = stats.get(lang, 0) + 1
        return stats

    def get_active_teachers(self) -> List[str]:
        """Return list of currently loaded teacher model names."""
        return list(self._teachers.keys())


def create_cross_lingual_router(
    preset: Optional[str] = None,
    language_mappings: Optional[Dict[str, str]] = None,
    default_teacher: Optional[str] = None,
    detector: Optional[LanguageDetector] = None,
    projection_enabled: bool = True,
    projection_hidden_dim: Optional[int] = None,
    teacher_weight: float = 0.5,
) -> CrossLingualTeacherRouter:
    """
    Factory function to create a CrossLingualTeacherRouter.

    Args:
        preset: Optional preset name from ChelationConfig.CROSS_LINGUAL_PRESETS.
        language_mappings: Optional dict mapping language codes to teacher model names.
        default_teacher: Fallback teacher model for unmapped languages.
        detector: Optional LanguageDetector instance.
        projection_enabled: Whether to auto-project dimension mismatches.
        projection_hidden_dim: Optional bottleneck dimension.
        teacher_weight: Default teacher weight for distillation.

    Returns:
        CrossLingualTeacherRouter instance.
    """
    if preset is not None:
        config = ChelationConfig.get_preset(preset, "cross_lingual")
        mappings = config.get("language_mappings", {})
        default = config.get("default_teacher", default_teacher)
        tw = config.get("teacher_weight", teacher_weight)
        mapping_obj = LanguageTeacherMapping(
            mappings=mappings,
            default_teacher=default or ChelationConfig.DEFAULT_TEACHER_MODEL,
        )
        return CrossLingualTeacherRouter(
            language_mapping=mapping_obj,
            detector=detector,
            projection_enabled=projection_enabled,
            projection_hidden_dim=projection_hidden_dim,
            teacher_weight=tw,
        )

    # Manual configuration
    mapping_obj = LanguageTeacherMapping(
        mappings=language_mappings or {},
        default_teacher=default_teacher or ChelationConfig.DEFAULT_TEACHER_MODEL,
    )
    return CrossLingualTeacherRouter(
        language_mapping=mapping_obj,
        detector=detector,
        projection_enabled=projection_enabled,
        projection_hidden_dim=projection_hidden_dim,
        teacher_weight=teacher_weight,
    )
