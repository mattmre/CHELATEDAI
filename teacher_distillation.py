"""
Teacher Distillation Module for ChelatedAI

Provides distillation helpers for offline teacher training and hybrid modes.
Uses local sentence-transformers model to avoid external API dependencies.
Supports batch-optimized encoding with chunked processing and parallel
multi-teacher encoding via ThreadPoolExecutor.
"""

import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from chelation_logger import get_logger
from config import ChelationConfig

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore


class DimensionProjection(nn.Module):
    """Projects teacher embeddings to student dimension space."""

    def __init__(self, teacher_dim, student_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(teacher_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, student_dim, bias=False),
            )
        else:
            self.projection = nn.Linear(teacher_dim, student_dim, bias=False)
        self._init_near_identity(teacher_dim, student_dim)

    def _init_near_identity(self, teacher_dim, student_dim):
        """Initialize near-identity for minimal disruption."""
        with torch.no_grad():
            if isinstance(self.projection, nn.Sequential):
                for module in self.projection:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        module.weight.mul_(0.001)
            else:
                min_dim = min(teacher_dim, student_dim)
                self.projection.weight.zero_()
                self.projection.weight[:min_dim, :min_dim] = (
                    torch.eye(min_dim) * 0.999
                )
                self.projection.weight += (
                    torch.randn_like(self.projection.weight) * 0.001
                )

    def forward(self, x):
        return self.projection(x)

    def project_numpy(self, embeddings):
        """Convenience method for numpy arrays."""
        tensor = torch.from_numpy(embeddings).float()
        with torch.no_grad():
            projected = self.forward(tensor)
        return projected.numpy()


class TeacherDistillationHelper:
    """
    Helper class for teacher distillation operations.

    Encapsulates teacher model loading, target generation, and blending logic.
    Supports batch-optimized encoding with configurable batch_size, chunked
    processing for large corpora, and eager model loading.
    """

    def __init__(self, teacher_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 projection_enabled: bool = True, projection_hidden_dim: Optional[int] = None,
                 batch_size: int = 64, eager_load: bool = False,
                 show_progress: bool = False, max_corpus_chunk: int = 10000):
        """
        Initialize teacher distillation helper.

        Args:
            teacher_model_name: HuggingFace model name for teacher
            projection_enabled: If True, auto-project when dimensions mismatch
            projection_hidden_dim: Optional bottleneck dimension for projection
            batch_size: Batch size for teacher model encoding (default: 64)
            eager_load: If True, load teacher model immediately (default: False)
            show_progress: If True, show progress bar during encoding (default: False)
            max_corpus_chunk: Maximum texts per encoding chunk (default: 10000)
        """
        self.logger = get_logger()
        self.teacher_model_name = teacher_model_name
        self.teacher_model = None
        self.teacher_dim = None
        self._projection_enabled = projection_enabled
        self._projection_hidden_dim = projection_hidden_dim
        self._projection = None
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.max_corpus_chunk = max_corpus_chunk

        self.logger.log_event(
            "distillation_init",
            f"Initializing teacher distillation with model: {teacher_model_name}",
            teacher_model=teacher_model_name,
            batch_size=batch_size,
            eager_load=eager_load,
            max_corpus_chunk=max_corpus_chunk,
        )

        if eager_load:
            self.load_teacher_model()
    
    def load_teacher_model(self):
        """Load teacher model lazily (only when needed)."""
        if self.teacher_model is not None:
            return
        
        try:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.log_event(
                "teacher_load",
                f"Loading teacher model on device: {device}",
                device=device
            )
            
            self.teacher_model = SentenceTransformer(
                self.teacher_model_name,
                device=device
            )
            self.teacher_dim = self.teacher_model.get_sentence_embedding_dimension()
            
            self.logger.log_event(
                "teacher_load_success",
                f"Teacher model loaded. Dimension: {self.teacher_dim}",
                teacher_dim=self.teacher_dim
            )
        except ImportError as e:
            self.logger.log_error(
                "teacher_load_failed",
                "sentence-transformers not available",
                exception=e
            )
            raise ImportError(
                "sentence-transformers required for teacher distillation. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            self.logger.log_error(
                "teacher_load_failed",
                f"Failed to load teacher model: {e}",
                exception=e
            )
            raise
    
    def _encode_chunk(self, chunk: List[str]) -> np.ndarray:
        """
        Encode a single chunk of texts using the teacher model.

        Args:
            chunk: List of text strings (should be <= max_corpus_chunk)

        Returns:
            Numpy array of embeddings for this chunk
        """
        return self.teacher_model.encode(
            chunk,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress,
            normalize_embeddings=True,
        )

    def get_teacher_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from teacher model with chunked encoding.

        Splits large text lists into max_corpus_chunk-sized chunks and encodes
        each independently. Per-chunk error handling ensures partial failures
        produce zero-vector fallbacks only for the failed chunk.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of teacher embeddings [batch_size, teacher_dim]
        """
        if not texts:
            return np.array([])

        self.load_teacher_model()

        # Split into chunks for memory-safe processing
        chunk_size = self.max_corpus_chunk
        if len(texts) <= chunk_size:
            # Fast path: single chunk (most common case)
            try:
                return self._encode_chunk(texts)
            except Exception as e:
                self.logger.log_error(
                    "teacher_embedding_failed",
                    f"Teacher embedding failed for {len(texts)} texts",
                    exception=e,
                    num_texts=len(texts),
                )
                return np.zeros((len(texts), self.teacher_dim))

        # Multi-chunk path
        all_embeddings = []
        num_chunks = (len(texts) + chunk_size - 1) // chunk_size
        failed_chunks = 0

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(texts))
            chunk = texts[start:end]

            try:
                chunk_embeddings = self._encode_chunk(chunk)
                all_embeddings.append(chunk_embeddings)
            except Exception as e:
                failed_chunks += 1
                self.logger.log_error(
                    "teacher_chunk_failed",
                    f"Chunk {chunk_idx + 1}/{num_chunks} failed ({len(chunk)} texts)",
                    exception=e,
                    chunk_index=chunk_idx,
                    chunk_size=len(chunk),
                )
                # Zero-vector fallback for this chunk only
                all_embeddings.append(np.zeros((len(chunk), self.teacher_dim)))

        if failed_chunks > 0:
            self.logger.log_event(
                "teacher_encoding_partial",
                f"Completed with {failed_chunks}/{num_chunks} chunk failures",
                level="WARNING",
                failed_chunks=failed_chunks,
                total_chunks=num_chunks,
            )

        return np.concatenate(all_embeddings, axis=0)
    
    def check_dimension_compatibility(self, student_dim: int) -> bool:
        """
        Check if teacher and student dimensions are compatible.
        
        Args:
            student_dim: Dimension of student embeddings
            
        Returns:
            True if compatible (same dimension), False otherwise
        """
        self.load_teacher_model()
        
        compatible = (self.teacher_dim == student_dim)
        
        if not compatible:
            self.logger.log_event(
                "dimension_mismatch",
                f"Teacher dim ({self.teacher_dim}) != Student dim ({student_dim})",
                level="WARNING",
                teacher_dim=self.teacher_dim,
                student_dim=student_dim
            )
        
        return compatible

    def _ensure_projection(self, student_dim):
        """Create projection layer if dimensions mismatch."""
        if self._projection is None:
            if self.teacher_dim != student_dim:
                self._projection = DimensionProjection(
                    self.teacher_dim, student_dim,
                    hidden_dim=self._projection_hidden_dim,
                )
                self.logger.log_event(
                    "projection_created",
                    f"Created dimension projection: {self.teacher_dim} -> {student_dim}",
                    teacher_dim=self.teacher_dim,
                    student_dim=student_dim,
                )

    def generate_distillation_targets(
        self,
        texts: List[str],
        current_embeddings: np.ndarray,
        teacher_weight: float = 1.0
    ) -> np.ndarray:
        """
        Generate distillation targets by blending teacher and current embeddings.
        
        Args:
            texts: Original text strings
            current_embeddings: Current student embeddings [N, dim]
            teacher_weight: Weight for teacher guidance (0=student only, 1=teacher only)
            
        Returns:
            Target embeddings [N, dim]
        """
        if teacher_weight == 0.0:
            # Pure student mode - no teacher guidance
            return current_embeddings.copy()
        
        # Get teacher embeddings
        teacher_embeds = self.get_teacher_embeddings(texts)
        
        if len(teacher_embeds) == 0 or len(teacher_embeds) != len(current_embeddings):
            self.logger.log_event(
                "distillation_target_fallback",
                "Teacher embedding mismatch, using student embeddings only",
                level="WARNING",
                teacher_size=len(teacher_embeds),
                student_size=len(current_embeddings)
            )
            return current_embeddings.copy()
        
        # Check dimension compatibility
        if teacher_embeds.shape[1] != current_embeddings.shape[1]:
            if self._projection_enabled:
                self._ensure_projection(current_embeddings.shape[1])
                if self._projection is not None:
                    teacher_embeds = self._projection.project_numpy(teacher_embeds)
                else:
                    return current_embeddings.copy()
            else:
                self.logger.log_event(
                    "dimension_mismatch_targets",
                    f"Cannot blend: teacher dim {teacher_embeds.shape[1]} != student dim {current_embeddings.shape[1]}",
                    level="ERROR",
                    teacher_dim=teacher_embeds.shape[1],
                    student_dim=current_embeddings.shape[1],
                )
                return current_embeddings.copy()
        
        # Blend: target = (1 - alpha) * student + alpha * teacher
        alpha = teacher_weight
        blended = (1 - alpha) * current_embeddings + alpha * teacher_embeds
        
        # Normalize to unit sphere (important for cosine similarity)
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)  # Avoid division by zero
        targets = blended / norms
        
        self.logger.log_event(
            "distillation_targets_generated",
            f"Generated {len(targets)} distillation targets",
            level="DEBUG",
            num_targets=len(targets),
            teacher_weight=teacher_weight
        )
        
        return targets
    
    def compute_alignment_metric(
        self,
        student_embeds: np.ndarray,
        teacher_embeds: np.ndarray
    ) -> float:
        """
        Compute average cosine similarity between student and teacher embeddings.
        
        Args:
            student_embeds: Student embeddings [N, dim]
            teacher_embeds: Teacher embeddings [N, dim]
            
        Returns:
            Average cosine similarity (0-1)
        """
        if len(student_embeds) == 0 or len(teacher_embeds) == 0:
            return 0.0
        
        if student_embeds.shape != teacher_embeds.shape:
            if (self._projection_enabled
                    and len(student_embeds.shape) == 2
                    and len(teacher_embeds.shape) == 2
                    and student_embeds.shape[0] == teacher_embeds.shape[0]
                    and student_embeds.shape[1] != teacher_embeds.shape[1]):
                self._ensure_projection(student_embeds.shape[1])
                if self._projection is not None:
                    teacher_embeds = self._projection.project_numpy(teacher_embeds)
                else:
                    return 0.0
            else:
                self.logger.log_event(
                    "alignment_shape_mismatch",
                    "Shape mismatch in alignment computation",
                    level="WARNING",
                    student_shape=student_embeds.shape,
                    teacher_shape=teacher_embeds.shape,
                )
                return 0.0
        
        # Compute row-wise cosine similarities
        student_norm = student_embeds / (np.linalg.norm(student_embeds, axis=1, keepdims=True) + 1e-9)
        teacher_norm = teacher_embeds / (np.linalg.norm(teacher_embeds, axis=1, keepdims=True) + 1e-9)
        
        similarities = np.sum(student_norm * teacher_norm, axis=1)
        avg_similarity = np.mean(similarities)
        
        return float(avg_similarity)


class EnsembleTeacherHelper:
    """Multi-teacher ensemble that duck-types TeacherDistillationHelper API.

    Supports parallel encoding of multiple teachers via ThreadPoolExecutor.
    """

    def __init__(self, teachers, weights=None, logger=None,
                 parallel_encoding=True, max_workers=4):
        """
        Args:
            teachers: List of TeacherDistillationHelper instances
            weights: Optional list of float weights (normalized internally)
            logger: Logger instance
            parallel_encoding: If True, encode teachers in parallel (default: True)
            max_workers: Max threads for parallel encoding (default: 4)
        """
        if not teachers:
            raise ValueError("At least one teacher is required")
        self.teachers = teachers
        self.logger = logger or get_logger()
        self.parallel_encoding = parallel_encoding
        self.max_workers = max_workers

        if weights is None:
            weights = [1.0 / len(teachers)] * len(teachers)
        total = sum(weights)
        self.weights = [w / total for w in weights]

        self._projection_enabled = True
        self._projections = {}  # teacher_idx -> DimensionProjection
        self.teacher_weight = 0.5

    def _encode_single_teacher(self, args):
        """Encode texts for a single teacher. Used by parallel path.

        Args:
            args: Tuple of (teacher_index, teacher, texts)

        Returns:
            Tuple of (teacher_index, embeddings_or_None)
        """
        i, teacher, texts = args
        try:
            embs = teacher.get_teacher_embeddings(texts)
            return (i, embs)
        except Exception as e:
            self.logger.log_error(
                "ensemble_teacher_failed",
                f"Teacher {i} encoding failed",
                exception=e,
                teacher_index=i,
            )
            return (i, np.array([]))

    def get_teacher_embeddings(self, texts, target_dim=None):
        """Get weighted average of teacher embeddings.

        When parallel_encoding is True and there are multiple teachers,
        uses ThreadPoolExecutor for concurrent encoding.
        """
        if self.parallel_encoding and len(self.teachers) > 1:
            return self._get_teacher_embeddings_parallel(texts, target_dim)
        return self._get_teacher_embeddings_sequential(texts, target_dim)

    def _get_teacher_embeddings_sequential(self, texts, target_dim=None):
        """Sequential encoding path (original behavior)."""
        all_embeddings = []
        for i, teacher in enumerate(self.teachers):
            embs = teacher.get_teacher_embeddings(texts)
            if len(embs) == 0:
                continue
            if target_dim is not None and embs.shape[-1] != target_dim:
                if i not in self._projections:
                    self._projections[i] = DimensionProjection(
                        embs.shape[-1], target_dim,
                    )
                embs = self._projections[i].project_numpy(embs)
            all_embeddings.append(embs * self.weights[i])
        if not all_embeddings:
            return np.array([])
        return sum(all_embeddings)

    def _get_teacher_embeddings_parallel(self, texts, target_dim=None):
        """Parallel encoding path using ThreadPoolExecutor."""
        work_items = [
            (i, teacher, texts) for i, teacher in enumerate(self.teachers)
        ]
        results = {}
        effective_workers = min(self.max_workers, len(self.teachers))

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            for i, embs in executor.map(self._encode_single_teacher, work_items):
                results[i] = embs

        all_embeddings = []
        for i in range(len(self.teachers)):
            embs = results.get(i, np.array([]))
            if len(embs) == 0:
                continue
            if target_dim is not None and embs.shape[-1] != target_dim:
                if i not in self._projections:
                    self._projections[i] = DimensionProjection(
                        embs.shape[-1], target_dim,
                    )
                embs = self._projections[i].project_numpy(embs)
            all_embeddings.append(embs * self.weights[i])
        if not all_embeddings:
            return np.array([])
        return sum(all_embeddings)

    def generate_distillation_targets(self, texts, student_embeddings,
                                      teacher_weight=None):
        """Generate targets using weighted ensemble."""
        if teacher_weight is None:
            teacher_weight = self.teacher_weight
        student_dim = student_embeddings.shape[-1]
        teacher_embs = self.get_teacher_embeddings(texts, target_dim=student_dim)
        if len(teacher_embs) == 0:
            return student_embeddings.copy()
        blended = teacher_weight * teacher_embs + (1 - teacher_weight) * student_embeddings
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        return blended / norms

    def compute_alignment_metric(self, student_embeds, teacher_embeds=None,
                                 texts=None):
        """Average alignment across all teachers."""
        alignments = []
        for teacher in self.teachers:
            alignment = teacher.compute_alignment_metric(
                student_embeds, teacher_embeds if teacher_embeds is not None
                else student_embeds,
            )
            alignments.append(alignment)
        return sum(a * w for a, w in zip(alignments, self.weights))


def create_distillation_helper(
    teacher_model_name: Optional[str] = None,
    teacher_models=None,
    projection_enabled: bool = True,
    projection_hidden_dim: Optional[int] = None,
    batch_size: int = 64,
    eager_load: bool = False,
    show_progress: bool = False,
    max_corpus_chunk: int = 10000,
    parallel_encoding: bool = True,
    max_workers: int = 4,
):
    """
    Factory function to create distillation helper.

    Args:
        teacher_model_name: Optional single teacher model name (uses default if None)
        teacher_models: Optional list of (model_name, weight) tuples for ensemble
        projection_enabled: If True, auto-project when dimensions mismatch
        projection_hidden_dim: Optional bottleneck dimension for projection
        batch_size: Batch size for teacher model encoding (default: 64)
        eager_load: If True, load teacher model immediately (default: False)
        show_progress: If True, show progress bar during encoding (default: False)
        max_corpus_chunk: Maximum texts per encoding chunk (default: 10000)
        parallel_encoding: If True, encode ensemble teachers in parallel (default: True)
        max_workers: Max threads for parallel ensemble encoding (default: 4)

    Returns:
        TeacherDistillationHelper or EnsembleTeacherHelper instance
    """
    if teacher_models and len(teacher_models) > 1:
        teachers = []
        weights = []
        for model_name, weight in teacher_models:
            helper = TeacherDistillationHelper(
                teacher_model_name=model_name,
                projection_enabled=projection_enabled,
                projection_hidden_dim=projection_hidden_dim,
                batch_size=batch_size,
                eager_load=eager_load,
                show_progress=show_progress,
                max_corpus_chunk=max_corpus_chunk,
            )
            teachers.append(helper)
            weights.append(weight)
        return EnsembleTeacherHelper(
            teachers, weights,
            parallel_encoding=parallel_encoding,
            max_workers=max_workers,
        )

    # Single teacher path
    model = teacher_model_name
    if model is None and teacher_models:
        model = teacher_models[0][0]
    if model is None:
        model = ChelationConfig.DEFAULT_TEACHER_MODEL

    return TeacherDistillationHelper(
        teacher_model_name=model,
        projection_enabled=projection_enabled,
        projection_hidden_dim=projection_hidden_dim,
        batch_size=batch_size,
        eager_load=eager_load,
        show_progress=show_progress,
        max_corpus_chunk=max_corpus_chunk,
    )


# Convenience function for hybrid target generation
def generate_hybrid_targets(
    texts: List[str],
    current_embeddings: np.ndarray,
    homeostatic_targets: np.ndarray,
    teacher_weight: float = 0.5,
    teacher_helper: Optional[TeacherDistillationHelper] = None
) -> np.ndarray:
    """
    Generate hybrid targets by blending homeostatic and teacher guidance.
    
    Args:
        texts: Original text strings
        current_embeddings: Current embeddings [N, dim]
        homeostatic_targets: Homeostatic push targets [N, dim]
        teacher_weight: Weight for teacher vs homeostatic (0=homeostatic only, 1=teacher only)
        teacher_helper: Optional pre-initialized helper (creates new if None)
        
    Returns:
        Hybrid target embeddings [N, dim]
    """
    if teacher_helper is None:
        teacher_helper = create_distillation_helper()
    
    if teacher_weight == 0.0:
        # Pure homeostatic mode
        return homeostatic_targets.copy()
    
    # Get teacher embeddings
    teacher_embeds = teacher_helper.get_teacher_embeddings(texts)
    
    if len(teacher_embeds) == 0 or teacher_embeds.shape != current_embeddings.shape:
        # Fallback to homeostatic only
        return homeostatic_targets.copy()
    
    # Blend: target = (1 - alpha) * homeostatic + alpha * teacher
    alpha = teacher_weight
    blended = (1 - alpha) * homeostatic_targets + alpha * teacher_embeds
    
    # Normalize
    norms = np.linalg.norm(blended, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    targets = blended / norms
    
    return targets
